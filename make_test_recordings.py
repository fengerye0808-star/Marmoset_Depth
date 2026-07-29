#!/usr/bin/env python3
"""
Generate synthetic labelled recordings, so the dataset -> training -> live
pipeline can be exercised before any real data exists.

    python make_test_recordings.py --out test_recordings --sessions 6
    python build_dataset.py --recordings test_recordings --out test_dataset
    python train_baseline.py --dataset test_dataset --epochs 10

It writes through the real RecordingSession, so the output format is exactly
what the camera produces. Each synthetic "expression" is a distinct facial
deformation that ramps up over its segment; every session gets a different
random face distance (140-165 mm), which is what verifies that the reference
normalization really does remove absolute distance.

This is a pipeline test, not a science dataset -- a model trained on it tells
you the plumbing works, nothing more.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import d405_recorder as rec

HERE = Path(__file__).resolve().parent


def face_depth(h, w, t, kind, ref_m, rng, clutter=True):
    """Synthetic head-fixed face at ref_m metres, deformed per `kind`.
    t in [0, 1] is progress through the segment, so deformations are dynamic.

    With clutter=True the scene also contains the things a real head-fixed rig
    puts in front of the camera -- the animal's torso, the headbar and a
    lickspout, all within the sensor's usable range. That is what a naive ROI
    estimate latches onto, so the synthetic data must contain it or the tests
    validate only an unrealistically clean scene.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    # face spans ~40% of the frame width, ~66% of its height, so the fitted ROI
    # is a genuine crop rather than the whole frame
    nx = (xx - cx) / (w * 0.115)
    ny = (yy - cy) / (h * 0.19)
    r2 = nx ** 2 + ny ** 2

    face = np.exp(-r2) * 0.028                      # base 28 mm face relief
    # Rise over the first quarter of the segment, then hold -- like a
    # stimulus-evoked expression that persists. A rise-and-fall pulse would
    # leave half of every segment indistinguishable from neutral, which caps
    # achievable accuracy for reasons that have nothing to do with the pipeline.
    ramp = float(min(1.0, max(0.0, t) / 0.25))

    if kind == "neutral":
        pass
    elif kind == "condition_1":                     # upper face pulls back
        face += ramp * 0.010 * np.exp(-((nx / 0.9) ** 2 + ((ny + 0.55) / 0.45) ** 2))
    elif kind == "condition_2":                     # lateral widening
        face += ramp * 0.009 * (np.exp(-(((nx - 0.7) / 0.4) ** 2 + (ny / 0.8) ** 2))
                                + np.exp(-(((nx + 0.7) / 0.4) ** 2 + (ny / 0.8) ** 2)))
    elif kind == "condition_3":                     # lower jaw drops forward
        face -= ramp * 0.012 * np.exp(-((nx / 0.7) ** 2 + ((ny - 0.6) / 0.5) ** 2))
    else:
        face += ramp * 0.006 * np.exp(-r2 * 0.5)

    depth_m = np.where(r2 < 3.0, ref_m - face, 0.55)     # 0.55 m = far wall

    if clutter:
        face_region = r2 < 3.0
        # torso sloping away in the bottom-right, 20 -> 35 cm
        torso = (yy > h * 0.72) & (xx > w * 0.55) & ~face_region
        depth_m[torso] = (0.20 + 0.15 * (xx[torso] / w))
        # headbar: a rigid post at 12 cm down the left edge
        headbar = (xx < w * 0.10) & ~face_region
        depth_m[headbar] = 0.12
        # lickspout at 9.5 cm, near the mouth but not touching the face
        spout = (((xx - w * 0.38) / (w * 0.022)) ** 2
                 + ((yy - h * 0.93) / (h * 0.05)) ** 2) < 1.0
        depth_m[spout & ~face_region] = 0.095

    depth_m += rng.normal(0, 0.0005, depth_m.shape)      # 0.5 mm sensor noise
    holes = rng.random(depth_m.shape) < 0.01             # stereo dropouts
    depth_m[holes] = 0.0
    return depth_m


def main():
    ap = argparse.ArgumentParser(description="Synthetic labelled recordings")
    ap.add_argument("--out", default=str(HERE / "test_recordings"))
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--seconds-per-label", type=float, default=1.0)
    ap.add_argument("--neutral-warmup", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=424)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-clutter", action="store_true",
                    help="omit the torso/headbar/spout (unrealistically clean)")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    labels = rec.load_labels()
    expressive = [l for l in labels if l not in ("unlabeled",)]
    if not expressive:
        sys.exit("labels.txt has no usable labels besides 'unlabeled'")
    # The baseline segment defines the reference; don't assume it is called
    # "neutral" just because the shipped labels.txt says so.
    base = "neutral" if "neutral" in expressive else expressive[0]
    if base != "neutral":
        print(f"note: no 'neutral' label in labels.txt, using '{base}' as the "
              f"baseline/reference segment")
    scale = 0.0001
    # Plausible D405 depth intrinsics: ~87 deg horizontal FOV, square pixels,
    # principal point at the image centre. Stored in the same place the real
    # recorder stores them so export_3d.py works on synthetic data unchanged.
    hfov_deg = 87.0
    fx = (args.width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
    meta = {
        "device": "Synthetic D405", "serial_number": "SYNTH",
        "firmware_version": "-",
        "stream": {"width": args.width, "height": args.height,
                   "fps": args.fps, "depth_format": "Z16"},
        "depth_scale_m_per_unit": scale,
        "depth_intrinsics": {
            "width": args.width, "height": args.height,
            "fx": round(fx, 3), "fy": round(fx, 3),
            "ppx": args.width / 2.0, "ppy": args.height / 2.0,
            "model": "synthetic_pinhole", "coeffs": [0.0] * 5,
        },
    }

    print(f"labels: {expressive}")
    n_per = max(2, int(round(args.seconds_per_label * args.fps)))
    n_warm = max(2, int(round(args.neutral_warmup * args.fps)))

    for s in range(args.sessions):
        rng = np.random.default_rng(args.seed + s)
        ref_m = float(rng.uniform(0.140, 0.165))     # different distance/session
        session = rec.RecordingSession(out_root, meta, labels=labels)
        session.log_event("synthetic_session", s)

        plan = [(base, n_warm)] + [(l, n_per) for l in expressive]
        trial = 0
        for kind, count in plan:
            label_idx = labels.index(kind)
            if kind != base:
                trial += 1
                session.log_event("trial_start", trial)
            session.log_event("label", kind)
            for i in range(count):
                depth_m = face_depth(args.height, args.width,
                                     i / max(1, count - 1), kind, ref_m, rng,
                                     clutter=not args.no_clutter)
                raw = np.clip(depth_m / scale, 0, 65535).astype(np.uint16)
                session.add_frame(raw, i * 1000.0 / args.fps, time.time(),
                                  label_idx, trial)
                while session.queue.qsize() > rec.QUEUE_SIZE // 2:
                    time.sleep(0.002)               # let the writer keep up
            if kind != base:
                session.log_event("trial_end", trial)
        session.stop()
        while session.state != "done":
            time.sleep(0.05)
        print(f"  session {s+1}/{args.sessions}: ref {ref_m*1000:.0f} mm  "
              f"{session.result_msg}")

    print(f"\nwrote {args.sessions} synthetic recordings to {out_root}\n"
          f"next:  python build_dataset.py --recordings {out_root} "
          f"--out test_dataset")


if __name__ == "__main__":
    main()
