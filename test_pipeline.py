#!/usr/bin/env python3
"""
Self-checks for the depth -> dataset -> model pipeline.

    python test_pipeline.py            # fast checks, no torch needed
    python test_pipeline.py --full     # also builds a dataset and trains 2 epochs

The important one is test_offline_live_parity: it proves that the tensor
live_infer.py feeds the network is bit-for-bit what build_dataset.py stored for
the same raw frames. If that ever fails, real-time accuracy will silently
diverge from the accuracy you measured while training.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import (PREPROC_VERSION, DEFAULT_PARAMS, DepthPreprocessor,
                           to_model_input, masked_resize, square_roi,
                           frame_stats, input_scale)

RESULTS = []


def check(name):
    def deco(fn):
        def wrapped(*a, **k):
            t0 = time.time()
            try:
                fn(*a, **k)
                RESULTS.append((name, True, f"{(time.time()-t0)*1000:.0f} ms"))
                print(f"  PASS  {name}")
            except Exception as e:
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
        return wrapped
    return deco


# ----------------------------------------------------- synthetic raw frames
def synth_scene(h=480, w=848, ref_m=0.15, bulge=0.0, scale=0.0001, seed=0,
                clutter=False, face=True):
    """A synthetic head-fixed scene at the recorder's real stream size.

    clutter=True adds the structures a real rig unavoidably places inside the
    sensor's range -- torso at 20-35 cm, headbar at 12 cm, lickspout at 9.5 cm.
    Returns (raw_uint16, face_mask).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # face spans ~40% of the frame width, ~66% of its height -- roughly a human
    # face at 15 cm through the D405's wide depth FOV
    nx, ny = (xx - w / 2) / (w * 0.115), (yy - h / 2) / (h * 0.19)
    r2 = nx ** 2 + ny ** 2
    relief = np.exp(-r2) * 0.028 + bulge * np.exp(
        -((nx / 0.7) ** 2 + ((ny - 0.6) / 0.5) ** 2))
    face_mask = (r2 < 3.0) & face
    depth_m = np.full((h, w), 0.55, np.float32)          # far wall
    depth_m[face_mask] = (ref_m - relief)[face_mask]

    if clutter:
        torso = (yy > h * 0.72) & (xx > w * 0.55) & ~face_mask
        depth_m[torso] = 0.20 + 0.15 * (xx[torso] / w)
        headbar = (xx < w * 0.10) & ~face_mask
        depth_m[headbar] = 0.12
        spout = (((xx - w * 0.38) / (w * 0.022)) ** 2
                 + ((yy - h * 0.93) / (h * 0.05)) ** 2) < 1.0
        depth_m[spout & ~face_mask] = 0.095

    depth_m = depth_m + rng.normal(0, 0.0004, depth_m.shape)
    depth_m[rng.random(depth_m.shape) < 0.01] = 0.0
    return np.clip(depth_m / scale, 0, 65535).astype(np.uint16), face_mask


def synth_raw(**kw):
    return synth_scene(**kw)[0]


# ----------------------------------------------------------------- unit tests
@check("masked_resize keeps NaN where there is no support")
def t_masked_resize():
    data = np.ones((64, 64), np.float32) * 5.0
    valid = np.zeros((64, 64), bool)
    valid[:32] = True
    out = masked_resize(data, valid, 16, 0.25)
    assert np.allclose(out[:8], 5.0), "valid half must keep its value"
    assert np.all(np.isnan(out[8:])), "unsupported half must be NaN"

    # Partial coverage: a valid value must survive at any support above the
    # threshold, and never be diluted toward the invalid filler.
    for frac, expect_valid in ((0.3, True), (0.5, True), (0.7, True),
                              (0.1, False)):
        v = np.zeros((64, 64), bool)
        n_cols = int(round(64 * frac))
        v[:, :n_cols] = True
        d = np.full((64, 64), 7.0, np.float32)
        o = masked_resize(d, v, 8, 0.25)
        col = o[:, 0]
        if expect_valid:
            assert np.isfinite(col).all(), f"coverage {frac} lost data"
            assert np.allclose(col[np.isfinite(col)], 7.0, atol=1e-4), \
                f"coverage {frac} diluted the value to {col[0]}"
        finite = np.isfinite(o)
        if finite.any():
            assert np.allclose(o[finite], 7.0, atol=1e-4), \
                "no output pixel may be a blend of valid and filler"


@check("square_roi stays inside the frame")
def t_square_roi():
    for bbox in [(0, 0, 10, 10), (400, 200, 424, 240), (-5, -5, 500, 500)]:
        x, y, s = square_roi(bbox, (240, 424), 1.35)
        assert x >= 0 and y >= 0, (x, y)
        assert x + s <= 424 and y + s <= 240, (x, y, s)
        assert s >= 8


@check("reference normalization removes absolute camera distance")
def t_distance_invariance():
    """Same face, two different distances -> same relative-depth output.
    This is what lets sessions and subjects be pooled."""
    outs = []
    for ref_m in (0.130, 0.170):
        frames = [synth_raw(ref_m=ref_m, seed=s) for s in range(10)]
        pre = DepthPreprocessor(0.0001).fit(frames)
        outs.append(pre.transform(synth_raw(ref_m=ref_m, seed=99)))
    a, b = outs
    both = np.isfinite(a) & np.isfinite(b)
    # a square ROI around an oval face is only ~half face, so ~0.4 is expected
    assert both.mean() > 0.35, \
        f"too few shared valid pixels ({both.mean():.2f})"
    diff = np.abs(a[both] - b[both])
    assert np.median(diff) < 1.0, (
        f"40 mm distance change leaked {np.median(diff):.2f} mm into the "
        f"normalized output")


@check("deformation survives normalization")
def t_deformation_signal():
    """A real facial bulge must NOT be normalized away."""
    frames = [synth_raw(ref_m=0.15, seed=s) for s in range(10)]
    pre = DepthPreprocessor(0.0001).fit(frames)
    neutral = pre.transform(synth_raw(ref_m=0.15, seed=99))
    bulged = pre.transform(synth_raw(ref_m=0.15, bulge=0.012, seed=99))
    both = np.isfinite(neutral) & np.isfinite(bulged)
    peak = np.abs(neutral[both] - bulged[both]).max()
    assert peak > 5.0, f"12 mm bulge only moved the output {peak:.2f} mm"


@check("ROI and reference ignore rig clutter (torso / headbar / spout)")
def t_clutter_rejected():
    """The regression test for the failure that destroys everything: if the ROI
    is the bbox of all in-range pixels, the reference lands on the torso, the
    face saturates at the clip rail, and expression signal becomes exactly
    zero while every summary still reports the session as healthy."""
    ref_m = 0.15
    frames, mask = [], None
    for s in range(25):
        raw, mask = synth_scene(ref_m=ref_m, clutter=True, seed=s)
        frames.append(raw)
    pre = DepthPreprocessor(0.0001).fit(frames)

    assert abs(pre.reference_mm - ref_m * 1000) < 20, (
        f"reference {pre.reference_mm:.0f} mm is not on the face "
        f"({ref_m*1000:.0f} mm) -- it locked onto clutter")

    x, y, s = pre.roi
    h, w = mask.shape
    assert s < min(h, w), f"ROI side {s} grew to the whole frame"
    inside = mask[y:y + s, x:x + s]
    assert inside.mean() > 0.35, (
        f"only {inside.mean()*100:.0f}% of the ROI is face -- ROI is too big")
    assert inside.sum() / mask.sum() > 0.9, (
        f"ROI covers only {inside.sum()/mask.sum()*100:.0f}% of the face")

    neutral = pre.transform(synth_raw(ref_m=ref_m, clutter=True, seed=99))
    st = frame_stats(neutral, pre.clip_mm)
    assert st["rail_frac"] < 0.02, (
        f"{st['rail_frac']*100:.0f}% of pixels are pinned at the clip limit")
    assert st["std_mm"] > 1.0, (
        f"relief is only {st['std_mm']:.2f} mm -- the frame is information-free")

    # a real 3 mm deformation must move the model input
    bulged = pre.transform(synth_raw(ref_m=ref_m, bulge=0.003, clutter=True,
                                     seed=99))
    both = np.isfinite(neutral) & np.isfinite(bulged)
    assert np.abs(neutral[both] - bulged[both]).max() > 1.0, (
        "a 3 mm facial bulge produced no change in the preprocessed output")
    xa = to_model_input(neutral[None], pre.reference_image,
                        input_scale(pre.params, "deform"), "deform")
    xb = to_model_input(bulged[None], pre.reference_image,
                        input_scale(pre.params, "deform"), "deform")
    assert np.abs(xa[0] - xb[0]).max() > 0.02, \
        "the deform-mode network input did not change either"


@check("explicit roi_override bypasses face finding and is validated")
def t_roi_override():
    """The escape hatch for rigs where the face cannot be isolated
    automatically -- e.g. a mouse, whose head and body are contiguous in
    depth."""
    frames = [synth_scene(clutter=True, seed=s)[0] for s in range(12)]
    h, w = frames[0].shape
    box = [w // 2 - 60, h // 2 - 60, 120]
    pre = DepthPreprocessor(0.0001, roi_override=box).fit(frames)
    assert list(pre.roi) == box, f"roi_override ignored: {pre.roi}"
    # A tight central box is dominated by the protruding nose, so the reference
    # sits in front of the 150 mm face plane. What must hold is that it is on
    # the face surface and inside the declared tolerance.
    assert 118 < pre.reference_mm < 152, \
        f"reference {pre.reference_mm:.0f} mm is not on the face surface"
    assert abs(pre.reference_mm / 1000 - pre.expected_distance_m) \
        <= pre.ref_tolerance_m
    out = pre.transform(frames[0])
    assert out.shape == (pre.out_size, pre.out_size)
    assert np.isfinite(out).mean() > 0.9, "explicit ROI should be all face"
    st = frame_stats(out, pre.clip_mm)
    assert st["std_mm"] > 1.0 and st["rail_frac"] < 0.02, \
        f"explicit ROI produced degenerate data: {st}"
    bulged = pre.transform(synth_scene(bulge=0.003, clutter=True, seed=0)[0])
    both = np.isfinite(out) & np.isfinite(bulged)
    assert np.abs(out[both] - bulged[both]).max() > 0.8, \
        "deformation not detectable through an explicit ROI"

    for bad in ([0, 0, 5], [w - 10, 0, 100], [0, 0, h + 10]):
        try:
            DepthPreprocessor(0.0001, roi_override=bad).fit(frames)
            raise AssertionError(f"roi_override {bad} should be rejected")
        except ValueError:
            pass


@check("a scene with no face never silently enters the dataset")
def t_fit_refuses_wrong_surface():
    """Either fit refuses outright, or the frames must fail the build's quality
    gates. What must never happen is flat clutter passing as a face."""
    import build_dataset as bd
    frames = [synth_scene(clutter=True, face=False, seed=s)[0]
              for s in range(12)]
    try:
        pre = DepthPreprocessor(0.0001).fit(frames)
    except ValueError as e:
        msg = str(e).lower()
        assert "cm" in msg or "flat" in msg, \
            f"error must explain what went wrong: {e}"
        return
    rel = pre.transform(frames[0])
    st = frame_stats(rel, pre.clip_mm)
    face = np.isfinite(pre.reference_image)
    coverage = float(np.isfinite(rel)[face].mean()) if face.any() else 0.0
    assert (st["std_mm"] < bd.MIN_STD_MM or st["rail_frac"] > bd.MAX_RAIL_FRAC
            or coverage < bd.MIN_FACE_COVERAGE), (
        f"a face-less scene was accepted at {pre.reference_mm:.0f} mm and "
        f"passed every quality gate: {st}, coverage {coverage:.2f}")


@check("clipping happens after the deform difference, not before")
def t_clip_after_deform():
    """A static surface near the clip limit must not swallow deformation."""
    clip = np.full((1, 8, 8), 39.0, np.float32)      # just inside +/-40
    ref = np.full((8, 8), 39.0, np.float32)
    moved = clip - 6.0                               # 6 mm of real motion
    scale = input_scale(DEFAULT_PARAMS, "deform")
    a = to_model_input(clip, ref, scale, "deform")
    b = to_model_input(moved, ref, scale, "deform")
    assert abs(float(a[0].mean())) < 1e-6, "neutral deform must be ~0"
    assert abs(float(b[0].mean()) - (-6.0 / scale)) < 1e-3, (
        f"6 mm of deformation became {float(b[0].mean())*scale:.2f} mm")


@check("deform mode marks reference holes invalid instead of faking depth")
def t_deform_nan_reference():
    clip = np.full((2, 8, 8), 5.0, np.float32)
    ref = np.full((8, 8), 1.0, np.float32)
    ref[0, 0] = np.nan                              # hole in the neutral face
    x = to_model_input(clip, ref, 20.0, "deform")
    assert x[1, 0, 0, 0] == 0.0, \
        "a pixel missing from the reference must be masked invalid"
    assert x[0, 0, 0, 0] == 0.0, \
        "a masked pixel must not carry absolute depth into a deform map"
    assert x[1, 0, 1, 1] == 1.0 and abs(x[0, 0, 1, 1] - 4.0 / 20.0) < 1e-6


@check("to_model_input ranges, mask channel, and deform mode")
def t_model_input():
    clip = np.full((4, 8, 8), 10.0, np.float32)
    clip[:, 0, 0] = np.nan
    clip[:, 1, 1] = 1e6                       # absurd value must clip
    x = to_model_input(clip, clip_mm_max=40.0)
    assert x.shape == (2, 4, 8, 8), x.shape
    assert x.dtype == np.float32
    assert -1.0 <= x[0].min() and x[0].max() <= 1.0
    assert x[1, 0, 0, 0] == 0.0, "mask must be 0 at NaN"
    assert x[0, 0, 0, 0] == 0.0, "depth must be 0 at NaN"
    assert x[1, 0, 2, 2] == 1.0, "mask must be 1 where valid"
    ref = np.full((8, 8), 10.0, np.float32)
    xd = to_model_input(clip, reference_image=ref, clip_mm_max=40.0,
                        mode="deform")
    assert abs(float(xd[0, 0, 3, 3])) < 1e-6, "deform of neutral must be ~0"


@check("preprocessor state_dict round-trips and version-checks")
def t_state_dict():
    frames = [synth_raw(seed=s) for s in range(8)]
    pre = DepthPreprocessor(0.0001).fit(frames)
    st = json.loads(json.dumps(pre.state_dict()))     # survives JSON
    back = DepthPreprocessor.from_state_dict(st)
    raw = synth_raw(seed=77)
    a, b = pre.transform(raw), back.transform(raw)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(a[np.isfinite(a)], b[np.isfinite(b)], atol=1e-4)
    assert back.roi == pre.roi and back.reference_mm == pre.reference_mm
    bad = dict(st, preproc_version="v0-bogus")
    try:
        DepthPreprocessor.from_state_dict(bad)
        raise AssertionError("stale preproc version must be rejected")
    except ValueError:
        pass


@check("transform before fit is refused")
def t_unfitted():
    try:
        DepthPreprocessor(0.0001).transform(synth_raw())
        raise AssertionError("transform() must require fit()")
    except RuntimeError:
        pass


@check("out-of-range subject gives a clear error, not garbage")
def t_out_of_range():
    far = np.full((480, 848), int(0.9 / 0.0001), np.uint16)   # 90 cm wall only
    try:
        DepthPreprocessor(0.0001).fit([far] * 5)
        raise AssertionError("expected a ValueError for an empty ROI")
    except ValueError as e:
        assert "cm" in str(e), f"error should tell the user the range: {e}"


@check("a mismatched frame size is refused rather than silently miscropped")
def t_shape_mismatch():
    pre = DepthPreprocessor(0.0001).fit([synth_raw(seed=s) for s in range(8)])
    try:
        pre.transform(np.zeros((240, 424), np.uint16))
        raise AssertionError("a different resolution must be refused")
    except ValueError as e:
        assert "shape" in str(e).lower()


@check("per-frame DC removal cancels a whole-face distance shift")
def t_per_frame_dc():
    frames = [synth_raw(ref_m=0.15, seed=s) for s in range(20)]
    base = DepthPreprocessor(0.0001, per_frame_dc="roi_median").fit(frames)
    a = base.transform(synth_raw(ref_m=0.150, seed=99))
    b = base.transform(synth_raw(ref_m=0.153, seed=99))   # 3 mm drift
    both = np.isfinite(a) & np.isfinite(b)
    assert np.abs(np.median(a[both] - b[both])) < 0.5, (
        f"3 mm drift left {np.median(a[both]-b[both]):.2f} mm of DC offset "
        f"even with per_frame_dc=roi_median")


# ---------------------------------------------------- full pipeline + parity
@check("offline dataset and live inference produce identical tensors")
def t_offline_live_parity(ds_dir, rec_dir):
    """The parity guarantee. Rebuild the live path from the raw PNGs and
    compare against what build_dataset.py stored."""
    from build_dataset import imread_u16, read_session, pick_reference_rows

    session = sorted(p.stem for p in (ds_dir / "sessions").glob("*.json"))[0]
    info = json.loads((ds_dir / "sessions" / f"{session}.json")
                      .read_text(encoding="utf-8"))
    stored = np.load(ds_dir / "sessions" / f"{session}.npy", mmap_mode="r")
    src = Path(info["source_dir"])
    meta, rows = read_session(src)

    # --- live path: fit on the same neutral frames, transform the same PNGs ---
    policy = json.loads((ds_dir / "dataset_spec.json")
                        .read_text(encoding="utf-8"))["reference_policy"]
    ref_rows, _, _ = pick_reference_rows(
        rows, policy["ref_label"], policy["ref_seconds"], info["fps"],
        policy["min_ref_frames"],
        policy["allow_session_start_reference"])
    assert ref_rows is not None, "reference policy rejected a built session"
    ref_frames = [imread_u16(src / r["filename"]) for r in ref_rows]
    ref_frames = [f for f in ref_frames if f is not None]
    live = DepthPreprocessor(
        info["preproc"]["depth_scale"],
        **info["preproc"]["params"]).fit(ref_frames)

    assert tuple(live.roi) == tuple(info["preproc"]["roi"]), (
        f"live ROI {live.roi} != dataset ROI {info['preproc']['roi']}")
    assert abs(live.reference_mm - info["preproc"]["reference_mm"]) < 1e-6, (
        f"live reference {live.reference_mm} != dataset "
        f"{info['preproc']['reference_mm']}")

    T = 16
    start = len(rows) // 3
    live_clip = np.stack([live.transform(imread_u16(src / rows[i]["filename"]))
                          for i in range(start, start + T)])
    offline_clip = np.asarray(stored[start:start + T], np.float32)

    assert np.array_equal(np.isnan(live_clip), np.isnan(offline_clip)), \
        "validity masks differ between offline and live"
    fin = np.isfinite(live_clip)
    # float16 storage is the only permitted difference
    assert np.allclose(live_clip[fin], offline_clip[fin], atol=0.05), \
        f"max diff {np.abs(live_clip[fin]-offline_clip[fin]).max():.4f} mm"

    offline_pre = DepthPreprocessor.from_state_dict(info["preproc"],
                                                    strict=False)
    for mode in ("absolute", "deform"):
        scale = input_scale(live.params, mode)
        xa = to_model_input(live_clip,
                            live.reference_image if mode == "deform" else None,
                            clip_mm_max=scale, mode=mode)
        xb = to_model_input(offline_clip,
                            offline_pre.reference_image
                            if mode == "deform" else None,
                            clip_mm_max=scale, mode=mode)
        assert np.allclose(xa, xb, atol=2e-3), \
            f"mode={mode} model inputs differ by {np.abs(xa-xb).max():.4f}"


@check("session splits are stable as the cohort grows")
def t_split_stability():
    """Adding recordings must never move an existing session to another split.
    A shuffle-and-partition scheme silently reassigns them, turning yesterday's
    train sessions into today's test set."""
    from build_dataset import assign_splits
    names = [f"2026-07-{d:02d}_10-00-00_dur10.0s" for d in range(1, 21)]

    # growing the cohort, carrying the previous assignment forward
    running = assign_splits(names[:6], 0.2, 0.2, 0)
    for n in range(7, 21):
        later = assign_splits(names[:n], 0.2, 0.2, 0, previous=running)
        for name, split in running.items():
            assert later[name] == split, (
                f"{name} moved from {split} to {later[name]} at {n} sessions")
        running = later

    counts = Counter(running.values())
    assert set(counts) == {"train", "val", "test"}, counts
    assert counts["train"] >= 0.5 * len(names), \
        f"train share too small: {counts}"

    # a session inserted out of order must not disturb the pinned ones
    extra = dict(running)
    with_new = assign_splits(names + ["2026-06-01_09-00-00_dur5.0s"],
                             0.2, 0.2, 0, previous=extra)
    for name, split in extra.items():
        assert with_new[name] == split, f"{name} moved when an older-named " \
                                        f"session was added"

    # a request of zero must create no such split
    z = assign_splits(names, 0.0, 0.2, 0)
    assert "val" not in set(z.values()), "--val-frac 0 still produced a val split"


@check("quality gates reject saturated / information-free sessions")
def t_quality_gates():
    import build_dataset as bd
    flat = np.zeros((32, 32), np.float32)            # no relief at all
    st = frame_stats(flat, 40.0)
    assert st["std_mm"] < bd.MIN_STD_MM, "a flat frame must fail the relief gate"
    railed = np.full((32, 32), -40.0, np.float32)    # everything at the rail
    st = frame_stats(railed, 40.0)
    assert st["rail_frac"] > bd.MAX_RAIL_FRAC, \
        "a fully saturated frame must fail the rail gate"

    good, _ = synth_scene(clutter=True, seed=3)
    pre = DepthPreprocessor(0.0001).fit(
        [synth_scene(clutter=True, seed=s)[0] for s in range(25)])
    rel = pre.transform(good)
    st = frame_stats(rel, pre.clip_mm)
    assert st["rail_frac"] <= bd.MAX_RAIL_FRAC and st["std_mm"] >= bd.MIN_STD_MM, \
        f"a healthy synthetic session must pass the gates, got {st}"
    # face coverage, not raw valid fraction, is the coverage gate: a square ROI
    # around an oval face is only ~half face even with a perfect measurement
    face = np.isfinite(pre.reference_image)
    coverage = float(np.isfinite(rel)[face].mean())
    assert coverage >= bd.MIN_FACE_COVERAGE, \
        f"healthy session had face coverage {coverage:.2f}"
    assert st["valid_frac"] < coverage, \
        "raw valid_frac should be lower than face coverage for a square ROI"


@check("dataset splits are grouped by session (no leakage)")
def t_split_grouping(ds_dir):
    spec = json.loads((ds_dir / "dataset_spec.json").read_text(encoding="utf-8"))
    groups = spec["split_sessions"]
    seen = {}
    for split, sessions in groups.items():
        for s in sessions:
            assert s not in seen, f"session {s} in both {seen[s]} and {split}"
            seen[s] = split
    import csv as _csv
    with open(ds_dir / "manifest_clips.csv", newline="",
              encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            assert seen[row["session"]] == row["split"], (
                f"clip split {row['split']} contradicts session split "
                f"{seen[row['session']]} for {row['session']}")


@check("dataset loader yields correctly shaped, finite batches")
def t_dataset_loader(ds_dir):
    from depth_dataset import DepthClipDataset
    ds = DepthClipDataset(ds_dir, split="train", mode="deform", augment=True)
    assert len(ds) > 0, "no training clips"
    spec = json.loads((ds_dir / "dataset_spec.json").read_text(encoding="utf-8"))
    S = int(spec["preproc_params"]["out_size"])
    T = int(spec["clip"]["length"])
    for i in (0, len(ds) // 2, len(ds) - 1):
        x, y = ds[i]
        x = np.asarray(x)
        assert x.shape == (2, T, S, S), x.shape
        assert np.isfinite(x).all(), "NaN/Inf reached the model input"
        assert -1.0001 <= x[0].min() and x[0].max() <= 1.0001
        assert int(y) < len(ds.labels)
    w = ds.class_weights()
    assert np.isfinite(w).all() and (w >= 0).all()
    counts = ds.class_counts()
    for c, wt in zip(counts, w):
        if c == 0:
            assert wt == 0.0, "absent class must get zero weight"


@check("3D export is metrically correct against known synthetic geometry")
def t_export_3d(rec_dir, tmp):
    """Validates the deprojection maths, not just that a file appears. The
    synthetic face has an exactly known pixel extent, so its physical size and
    aspect ratio in the exported mesh are checkable numbers."""
    import subprocess
    from export_3d import read_ply
    here = Path(__file__).resolve().parent
    session = sorted(d for d in rec_dir.iterdir() if d.is_dir())[0]
    out = tmp / "face.ply"
    r = subprocess.run(
        [sys.executable, "export_3d.py", "--recording", str(session),
         "--label", "neutral", "--average", "--mesh", "--out", str(out),
         "--preview", str(tmp / "face.png")],
        cwd=here, capture_output=True, text=True)
    assert r.returncode == 0, f"export_3d failed:\n{r.stdout}\n{r.stderr}"
    assert out.exists() and (tmp / "face.png").exists()

    verts, faces = read_ply(out)
    assert len(verts) > 1000, f"only {len(verts)} vertices"
    assert faces is not None and len(faces) > 1000
    assert faces.min() >= 0 and faces.max() < len(verts), \
        "mesh references vertices that do not exist"
    assert np.isfinite(verts).all(), "NaN reached the exported geometry"

    ext = verts.max(axis=0) - verts.min(axis=0)
    # make_test_recordings' face spans sqrt(3)*0.115*W x sqrt(3)*0.19*H pixels
    # at 140-165 mm, through fx = (W/2)/tan(43.5 deg).
    assert 95 < ext[0] < 140, f"face width {ext[0]:.0f} mm is not plausible"
    assert 88 < ext[1] < 130, f"face height {ext[1]:.0f} mm is not plausible"
    assert 18 < ext[2] < 38, f"face relief {ext[2]:.0f} mm is not plausible"
    # aspect ratio is scale-free, so it isolates the deprojection itself
    aspect = ext[0] / ext[1]
    assert abs(aspect - 169 / 158) < 0.12, \
        f"width/height {aspect:.3f} does not match the known 1.070"

    # metres must be exactly a 1000x rescale of millimetres
    out_m = tmp / "face_m.ply"
    r = subprocess.run(
        [sys.executable, "export_3d.py", "--recording", str(session),
         "--label", "neutral", "--average", "--mesh", "--units", "m",
         "--out", str(out_m)], cwd=here, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    v_m, _ = read_ply(out_m)
    ext_m = (v_m.max(axis=0) - v_m.min(axis=0)) * 1000.0
    assert np.allclose(ext, ext_m, rtol=0.02), \
        f"mm {ext} and m {ext_m} disagree"


@check("checkpoint round-trips and refuses a stale preproc version")
def t_checkpoint(model_path):
    import torch
    from live_infer import load_model
    model, ck = load_model(model_path, torch.device("cpu"))
    S = int(ck["preproc_params"]["out_size"])
    T = int(ck["clip_len"])
    x = np.zeros((1, 2, T, S, S), np.float32)
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(x)), 1)[0].numpy()
    assert p.shape == (len(ck["labels"]),), p.shape
    assert abs(p.sum() - 1) < 1e-4
    assert ck["preproc_version"] == PREPROC_VERSION

    bad = dict(torch.load(model_path, map_location="cpu", weights_only=False),
               preproc_version="v0-bogus")
    tmp = Path(model_path).with_name("stale_test.pt")
    torch.save(bad, tmp)
    try:
        load_model(tmp, torch.device("cpu"))
        raise AssertionError("stale checkpoint must be refused")
    except SystemExit:
        pass
    finally:
        tmp.unlink(missing_ok=True)


def run_full(tmp):
    """Generate recordings -> build -> train 2 epochs -> parity checks."""
    import subprocess
    here = Path(__file__).resolve().parent
    rec_dir, ds_dir, model_dir = tmp / "rec", tmp / "ds", tmp / "models"

    def run(*args):
        r = subprocess.run([sys.executable, *args], cwd=here,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            raise AssertionError(f"{args[0]} failed (exit {r.returncode})")
        return r.stdout

    print("\n  (generating 4 synthetic sessions with rig clutter ...)")
    run("make_test_recordings.py", "--out", str(rec_dir), "--sessions", "4",
        "--seconds-per-label", "0.7")
    print("  (building dataset ...)")
    run("build_dataset.py", "--recordings", str(rec_dir), "--out", str(ds_dir))
    t_offline_live_parity(ds_dir, rec_dir)
    t_split_grouping(ds_dir)
    t_dataset_loader(ds_dir)
    t_export_3d(rec_dir, tmp)
    print("  (training 2 epochs ...)")
    run("train_baseline.py", "--dataset", str(ds_dir), "--out", str(model_dir),
        "--epochs", "2")
    t_checkpoint(model_dir / "expression_model.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also build a dataset and train (slower, needs torch)")
    ap.add_argument("--keep", action="store_true", help="keep temp artifacts")
    args = ap.parse_args()

    print(f"preprocessing version {PREPROC_VERSION}, params {DEFAULT_PARAMS}")
    print("\nunit checks")
    t_masked_resize()
    t_square_roi()
    t_clutter_rejected()
    t_roi_override()
    t_fit_refuses_wrong_surface()
    t_distance_invariance()
    t_deformation_signal()
    t_clip_after_deform()
    t_deform_nan_reference()
    t_model_input()
    t_per_frame_dc()
    t_state_dict()
    t_unfitted()
    t_out_of_range()
    t_shape_mismatch()
    t_split_stability()
    t_quality_gates()

    if args.full:
        tmp = Path(tempfile.mkdtemp(prefix="d405_test_"))
        print(f"\nfull pipeline checks in {tmp}")
        try:
            run_full(tmp)
        finally:
            if args.keep:
                print(f"  artifacts kept in {tmp}")
            else:
                shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("\n(run with --full to also test dataset building, parity, "
              "and training)")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    for name, ok, note in RESULTS:
        if not ok:
            print(f"  FAILED {name}: {note}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
