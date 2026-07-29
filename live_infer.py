#!/usr/bin/env python3
"""
Real-time facial-expression readout from the live D405 depth stream.

    python live_infer.py                                   # models/expression_model.pt
    python live_infer.py --model models/expression_model.pt --mock
    python live_infer.py --log predictions.csv

Keeps the same camera source and the same DepthPreprocessor / to_model_input
calls used to build the training data, so what the network sees at run time is
what it was trained on.

Startup does a short WARM-UP: hold the subject still and neutral while it
estimates the face ROI and the neutral reference depth for this session (the
same thing build_dataset.py does offline from the recording's neutral period).
Press W to redo the warm-up if the subject or rig moves.

Keys:  Q/Esc quit    W re-run warm-up    SPACE pause/resume
"""

import argparse
import collections
import csv
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except ImportError:
    raise SystemExit("live inference needs PyTorch:\n"
                     "  pip install torch --index-url "
                     "https://download.pytorch.org/whl/cpu")

from d405_recorder import MockSource, RealSenseSource
from preprocessing import (DepthPreprocessor, input_scale, to_model_input,
                           PREPROC_VERSION)
from train_baseline import DepthExpressionNet

HERE = Path(__file__).resolve().parent
PANEL = 320


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("preproc_version") != PREPROC_VERSION:
        raise SystemExit(
            f"checkpoint was trained with preprocessing "
            f"{ck.get('preproc_version')} but this code is {PREPROC_VERSION}. "
            f"Retrain or check out the matching version -- running anyway "
            f"would feed the network inputs it never saw.")
    arch = ck["arch"]
    model = DepthExpressionNet(arch["n_classes"], in_ch=arch["in_ch"],
                               width=arch["width"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck


def colorize_rel(img, clip_mm):
    """Relative-depth image (mm, NaN invalid) -> BGR for display."""
    v = np.clip((np.nan_to_num(img, nan=0.0) / clip_mm + 1) / 2, 0, 1)
    col = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)
    col[~np.isfinite(img)] = 0
    return cv2.resize(col, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def draw_bars(probs, labels, width, height, smoothed=None):
    img = np.full((height, width, 3), 24, np.uint8)
    n = len(labels)
    row = max(18, min(34, height // max(1, n)))
    for i, (lab, pr) in enumerate(zip(labels, probs)):
        y = 6 + i * row
        if y + row > height:
            break
        top = int(np.argmax(probs)) == i
        bar = int((width - 150) * float(pr))
        cv2.rectangle(img, (140, y + 3), (140 + bar, y + row - 6),
                      (90, 200, 110) if top else (90, 90, 100), -1)
        cv2.putText(img, f"{lab[:16]}", (8, y + row - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255) if top else (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(img, f"{pr*100:4.0f}%", (width - 52, y + row - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255) if top else (150, 150, 150), 1, cv2.LINE_AA)
    return img


def warm_up(source, n_frames, preproc_params, depth_scale, status_cb):
    """Collect neutral frames and fit the session ROI + reference depth."""
    frames = []
    t_start = time.time()
    while len(frames) < n_frames and time.time() - t_start < 20:
        f = source.read()
        if f is None:
            continue
        frames.append(f.depth)
        status_cb(f, len(frames), n_frames)
    if len(frames) < 3:
        raise RuntimeError("warm-up got no frames from the camera")
    pre = DepthPreprocessor(depth_scale, **preproc_params)
    pre.fit(frames)
    return pre


def main():
    ap = argparse.ArgumentParser(description="Real-time depth expression readout")
    ap.add_argument("--model", default=str(HERE / "models" /
                                           "expression_model.pt"))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--warmup-frames", type=int, default=45,
                    help="neutral frames used to fit ROI + reference")
    ap.add_argument("--every", type=int, default=2,
                    help="run the network every N frames (1 = every frame)")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="EMA factor on probabilities, 0 = off")
    ap.add_argument("--log", default=None,
                    help="append per-prediction rows to this CSV")
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"no model at {args.model} -- train one first:\n"
                         f"  python build_dataset.py && python train_baseline.py")
    device = torch.device(args.device)
    model, ck = load_model(args.model, device)
    labels = ck["labels"]
    clip_len = int(ck["clip_len"])
    mode = ck.get("mode", "absolute")
    pparams = dict(ck["preproc_params"])
    # input_scale is shared with the training dataset, so the [-1,1] mapping
    # here is identical to the one the model was trained under.
    clip_mm = input_scale(pparams, mode)
    print(f"model: {Path(args.model).name}  labels={labels}  "
          f"clip_len={clip_len}  mode={mode}  device={device}")

    source = MockSource() if args.mock else RealSenseSource()
    win = "D405 live expression  |  Q quit   W re-warm-up   SPACE pause"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def warm_status(f, k, n):
        left = f.color if f.color is not None else np.zeros(
            (PANEL, PANEL, 3), np.uint8)
        left = cv2.resize(left, (PANEL, PANEL))
        panel = np.hstack([left, cv2.resize(f.depth_vis, (PANEL, PANEL))])
        cv2.putText(panel, f"WARM-UP {k}/{n}: hold subject still & NEUTRAL",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(panel, f"WARM-UP {k}/{n}: hold subject still & NEUTRAL",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 230, 250),
                    1, cv2.LINE_AA)
        cv2.imshow(win, panel)
        cv2.waitKey(1)

    try:
        pre = warm_up(source, args.warmup_frames, pparams,
                      source.depth_scale, warm_status)
    except (RuntimeError, ValueError) as e:
        source.stop()
        cv2.destroyAllWindows()
        raise SystemExit(f"warm-up failed: {e}")
    print(f"warm-up done: ROI={pre.roi} reference={pre.reference_mm:.0f} mm")

    buf = collections.deque(maxlen=clip_len)
    probs = np.full(len(labels), 1.0 / len(labels), np.float32)
    smoothed = probs.copy()
    have_pred = False      # so the EMA starts at the first real prediction
    net_input = None       # last tensor actually fed to the model, for display
    log_writer = log_file = None
    if args.log:
        log_file = open(args.log, "a", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        if log_file.tell() == 0:
            log_writer.writerow(["host_time_unix", "iso", "top_label",
                                 "confidence"] + [f"p_{l}" for l in labels])

    paused = False
    frame_i = 0
    infer_ms = 0.0
    fps_times = collections.deque(maxlen=30)
    try:
        while True:
            f = source.read()
            if f is None:
                continue
            fps_times.append(time.time())
            if not paused:
                rel = pre.transform(f.depth)
                buf.append(rel)
                frame_i += 1
                if len(buf) == clip_len and frame_i % max(1, args.every) == 0:
                    clip = np.stack(buf)
                    x = to_model_input(
                        clip,
                        reference_image=pre.reference_image
                        if mode == "deform" else None,
                        clip_mm_max=clip_mm, mode=mode)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        logits = model(torch.from_numpy(x[None]).to(device))
                        probs = torch.softmax(logits, 1)[0].cpu().numpy()
                    infer_ms = (time.perf_counter() - t0) * 1000
                    net_input = x
                    a = args.smooth
                    if a > 0 and have_pred:
                        smoothed = a * smoothed + (1 - a) * probs
                    else:
                        smoothed = probs.copy()   # don't blend with the prior
                    have_pred = True
                    if log_writer:
                        top = int(np.argmax(smoothed))
                        log_writer.writerow(
                            [f"{f.host_time:.6f}",
                             time.strftime("%Y-%m-%dT%H:%M:%S",
                                           time.localtime(f.host_time)),
                             labels[top], f"{smoothed[top]:.4f}"]
                            + [f"{p:.4f}" for p in smoothed])
                        log_file.flush()   # survive a hard kill / power cut

            # ---- display: color | live depth | network input | probabilities
            left = f.color if f.color is not None else np.zeros(
                (PANEL, PANEL, 3), np.uint8)
            left = cv2.resize(left, (PANEL, PANEL))
            x0, y0, s = pre.roi
            h, w = f.depth.shape
            sx, sy = PANEL / w, PANEL / h
            cv2.rectangle(left, (int(x0 * sx), int(y0 * sy)),
                          (int((x0 + s) * sx), int((y0 + s) * sy)),
                          (80, 220, 240), 1)
            mid = cv2.resize(f.depth_vis, (PANEL, PANEL))
            if net_input is not None:
                # show exactly what the model saw, not the pre-deform frame
                last = net_input[0, -1] * clip_mm
                shown = np.where(net_input[1, -1] > 0, last, np.nan)
            else:
                shown = np.full((pre.out_size, pre.out_size), np.nan,
                                np.float32)
            net_view = colorize_rel(shown, clip_mm)
            cv2.putText(net_view, f"network input ({mode})", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)
            bars = draw_bars(smoothed, labels, 300, PANEL)

            top = int(np.argmax(smoothed))
            ready = have_pred and len(buf) == clip_len
            banner = np.full((44, PANEL * 3 + 300, 3), 18, np.uint8)
            txt = (f"{labels[top]}   {smoothed[top]*100:.0f}%" if ready
                   else f"filling clip buffer {len(buf)}/{clip_len} ...")
            cv2.putText(banner, txt, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (90, 240, 120) if ready else (150, 150, 150), 2,
                        cv2.LINE_AA)
            fps = 0.0
            if len(fps_times) > 1:
                span = fps_times[-1] - fps_times[0]
                fps = (len(fps_times) - 1) / span if span > 0 else 0.0
            status = f"{fps:4.1f} fps   infer {infer_ms:4.1f} ms   " \
                     f"ref {pre.reference_mm:.0f} mm"
            if paused:
                status += "   [PAUSED]"
            cv2.putText(banner, status, (banner.shape[1] - 340, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1,
                        cv2.LINE_AA)
            frame = np.vstack([banner,
                               np.hstack([left, mid, net_view, bars])])
            cv2.imshow(win, frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q"), 27):
                break
            if k in (ord("w"), ord("W")):
                buf.clear()
                have_pred, net_input = False, None
                try:
                    pre = warm_up(source, args.warmup_frames, pparams,
                                  source.depth_scale, warm_status)
                    print(f"re-warmed: ROI={pre.roi} "
                          f"reference={pre.reference_mm:.0f} mm")
                except (RuntimeError, ValueError) as e:
                    # keep the previous fit rather than killing the session
                    print(f"re-warm-up failed, keeping the previous "
                          f"ROI/reference: {e}")
            if k == 32:
                paused = not paused
                # a clip must not straddle the pause gap
                buf.clear()
                have_pred, net_input = False, None
    except KeyboardInterrupt:
        pass
    finally:
        if log_file:
            log_file.close()
        source.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
