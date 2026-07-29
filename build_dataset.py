#!/usr/bin/env python3
"""
Turn raw D405 recordings into a training-ready dataset.

    python build_dataset.py                 # build/update dataset/ from recordings/
    python build_dataset.py --rebuild       # ignore the cache, redo everything
    python build_dataset.py --clip-len 24 --stride 6

For each recording it fits the canonical DepthPreprocessor (see
preprocessing.py) on that session's neutral warm-up period, applies it to
every frame, and stores the result as one float16 array per session. It is
INCREMENTAL: sessions already built with the same preprocessing version,
parameters, frame count and labels are skipped, so you can re-run it after
every recording day. Editing labels in frame_timestamps.csv triggers a cheap
metadata-only refresh rather than a full re-decode.

Sessions are REFUSED (not silently included) when the depth is saturated,
information-free, or the reference could not be fitted at the expected
distance -- the failure modes that otherwise look healthy in every summary.

Output layout
-------------
dataset/
  dataset_spec.json     preprocessing params, label map, splits, QC stats
  manifest_frames.csv   one row per frame  (session, label, trial, split, ...)
  manifest_clips.csv    one row per training clip (session, start, len, label)
  sessions/
    <session>.npy       (N, S, S) float16 relative depth in mm, NaN = invalid
    <session>.json      per-frame labels/trials/timestamps + preproc state
  qc/
    <session>.png       visual check: reference face, sample frames, ROI

The clip index in manifest_clips.csv is what you actually train on -- depth
frames alone don't carry expression dynamics, so samples are short windows.
Splits are grouped BY SESSION so no session appears in both train and test
(frames within a session are far too correlated to split randomly).
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from preprocessing import (PREPROC_VERSION, DEFAULT_PARAMS, PER_FRAME_DC_CHOICES,
                           DepthPreprocessor, frame_stats)

HERE = Path(__file__).resolve().parent
UNLABELED = "unlabeled"

# Quality gates. A session failing these is excluded from the dataset unless
# --allow-low-quality is passed.
MAX_RAIL_FRAC = 0.02        # fraction of pixels pinned at the clip limit
MIN_STD_MM = 0.5            # spatial relief; below this the frame is flat
# Coverage is measured against the NEUTRAL FACE FOOTPRINT, not the whole ROI:
# a square ROI around an oval face is only ~50% face even when the measurement
# is perfect, so a raw valid-pixel fraction is not a quality signal.
MIN_FACE_COVERAGE = 0.5     # refuse below this
WARN_FACE_COVERAGE = 0.8    # warn below this
DRIFT_WARN_MM = 2.0         # per-label DC spread that suggests a drift shortcut


def imread_u16(path):
    """Unicode-path-safe 16-bit PNG read."""
    try:
        buf = np.frombuffer(Path(path).read_bytes(), np.uint8)
    except OSError:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 2:
        return None
    return img


def read_session(rec_dir):
    """Read a recording folder -> (meta, rows). rows are dicts from
    frame_timestamps.csv with label/trial filled in for old formats."""
    meta_path = rec_dir / "metadata.json"
    csv_path = rec_dir / "frame_timestamps.csv"
    if not meta_path.exists() or not csv_path.exists():
        return None, None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ! unreadable metadata.json ({e})")
        return None, None
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "frame_index": int(row["frame_index"]),
                "filename": row["filename"],
                "label": row.get("label") or UNLABELED,
                "trial_id": int(row.get("trial_id") or 0),
                "device_timestamp_ms": row.get("device_timestamp_ms", ""),
                "host_time_unix": row.get("host_time_unix", ""),
            })
    return meta, rows


def label_digest(rows):
    """Fingerprint of the label/trial sequence, so edited labels invalidate the
    cached metadata (but not the pixel array, which labels don't affect)."""
    payload = json.dumps([(r["label"], r["trial_id"]) for r in rows])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def frame_times_ms(rows):
    """Best available per-frame clock, in ms. Camera clock preferred."""
    out = []
    for r in rows:
        t = None
        try:
            t = float(r["device_timestamp_ms"])
        except (TypeError, ValueError):
            try:
                t = float(r["host_time_unix"]) * 1000.0
            except (TypeError, ValueError):
                t = None
        out.append(t)
    return out


def pick_reference_rows(rows, ref_label, ref_seconds, fps, min_ref_frames,
                        allow_fallback):
    """Frames used to fit the ROI + reference depth.

    Requires a CONTIGUOUS run of `ref_label` (normally 'neutral') at least
    min_ref_frames long. A reference built from an arbitrary expression, or
    from poses minutes apart, offsets the whole session's deform space by that
    expression -- so this refuses rather than guesses, unless the caller
    explicitly allows the session-start fallback.
    """
    best, run = [], []
    for r in rows:
        if r["label"] == ref_label:
            run.append(r)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []
    n_want = max(min_ref_frames, int(round(ref_seconds * fps)))
    if len(best) >= min_ref_frames:
        return best[:n_want], f"label:{ref_label}", len(best)
    if allow_fallback:
        return rows[:n_want], "session_start", 0
    return None, (f"no contiguous run of '{ref_label}' >= {min_ref_frames} "
                  f"frames (longest was {len(best)})"), len(best)


def build_session(rec_dir, out_dir, args, preproc_overrides):
    name = rec_dir.name
    meta, rows = read_session(rec_dir)
    if meta is None:
        print(f"  skip {name}: not a complete recording")
        return None
    if not rows:
        print(f"  skip {name}: no frames")
        return None

    depth_scale = float(meta.get("camera", {})
                        .get("depth_scale_m_per_unit", 0.0001))
    fps = float(meta.get("camera", {}).get("stream", {}).get("fps", 30)) or 30.0
    params = {**DEFAULT_PARAMS, **preproc_overrides}
    digest = label_digest(rows)
    cache_key = {
        "preproc_version": PREPROC_VERSION, "params": params,
        "n_frames": len(rows), "depth_scale": depth_scale,
        "ref_label": args.ref_label, "ref_seconds": args.ref_seconds,
        "min_ref_frames": args.min_ref_frames,
        "allow_session_start_reference": args.allow_session_start_reference,
    }

    npy_path = out_dir / "sessions" / f"{name}.npy"
    json_path = out_dir / "sessions" / f"{name}.json"
    if not args.rebuild and npy_path.exists() and json_path.exists():
        try:
            cached = json.loads(json_path.read_text(encoding="utf-8"))
            if cached.get("cache_key") == cache_key \
                    and cached.get("array_bytes") == npy_path.stat().st_size:
                if cached.get("label_digest") == digest:
                    print(f"  cached {name} ({len(rows)} frames)")
                    return cached
                # Labels changed but pixels did not: refresh metadata only.
                cached["labels"] = [r["label"] for r in rows]
                cached["trial_ids"] = [r["trial_id"] for r in rows]
                cached["label_digest"] = digest
                json_path.write_text(json.dumps(cached), encoding="utf-8")
                print(f"  relabeled {name} ({len(rows)} frames, "
                      f"array reused)")
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    ref_rows, ref_source, run_len = pick_reference_rows(
        rows, args.ref_label, args.ref_seconds, fps, args.min_ref_frames,
        args.allow_session_start_reference)
    if ref_rows is None:
        print(f"  SKIP {name}: {ref_source}.\n"
              f"       Record ~{args.ref_seconds:.0f} s of '{args.ref_label}' "
              f"at the start of each session, or pass "
              f"--allow-session-start-reference to use the first frames "
              f"regardless of label.")
        return None

    ref_frames = [imread_u16(rec_dir / r["filename"]) for r in ref_rows]
    ref_frames = [f for f in ref_frames if f is not None]
    if len(ref_frames) < args.min_ref_frames:
        print(f"  SKIP {name}: only {len(ref_frames)} readable reference "
              f"frames, need {args.min_ref_frames}")
        return None

    pre = DepthPreprocessor(depth_scale, **preproc_overrides)
    try:
        pre.fit(ref_frames)
    except ValueError as e:
        print(f"  SKIP {name}: {e}")
        return None

    S = pre.out_size
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove the old json FIRST so an interrupted build cannot leave a stale
    # json next to a half-written array and be accepted by the cache.
    json_path.unlink(missing_ok=True)
    arr = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.float16,
                                    shape=(len(rows), S, S))
    face_mask = np.isfinite(pre.reference_image)
    n_face = int(face_mask.sum())
    per_frame = []
    missing = 0
    for i, r in enumerate(rows):
        img = imread_u16(rec_dir / r["filename"])
        if img is None:
            arr[i] = np.nan
            missing += 1
            per_frame.append({"valid_frac": 0.0, "rail_frac": 0.0,
                              "std_mm": 0.0, "median_mm": float("nan"),
                              "face_coverage": 0.0})
            continue
        try:
            rel = pre.transform(img)
        except ValueError as e:
            print(f"  SKIP {name}: frame {i}: {e}")
            del arr
            npy_path.unlink(missing_ok=True)
            return None
        arr[i] = rel.astype(np.float16)
        st = frame_stats(rel, pre.clip_mm)
        st["face_coverage"] = (float(np.isfinite(rel)[face_mask].mean())
                               if n_face else 0.0)
        per_frame.append(st)
    arr.flush()
    del arr

    valid_frac = float(np.mean([s["valid_frac"] for s in per_frame]))
    rail_frac = float(np.mean([s["rail_frac"] for s in per_frame]))
    std_mm = float(np.median([s["std_mm"] for s in per_frame]))
    coverage = float(np.mean([s["face_coverage"] for s in per_frame]))

    # Per-label DC: if these separate, slow depth drift is a shortcut the model
    # can exploit instead of learning facial deformation.
    by_label = defaultdict(list)
    for r, st in zip(rows, per_frame):
        if np.isfinite(st["median_mm"]):
            by_label[r["label"]].append(st["median_mm"])
    label_dc = {k: round(float(np.mean(v)), 3) for k, v in by_label.items()}
    dc_spread = (round(max(label_dc.values()) - min(label_dc.values()), 3)
                 if len(label_dc) > 1 else 0.0)

    problems = []
    if rail_frac > MAX_RAIL_FRAC:
        problems.append(
            f"{rail_frac*100:.1f}% of pixels are pinned at the "
            f"+/-{pre.clip_mm:.0f} mm limit (max {MAX_RAIL_FRAC*100:.0f}%) -- "
            f"the reference is likely on the wrong surface")
    if std_mm < MIN_STD_MM:
        problems.append(
            f"depth relief is only {std_mm:.2f} mm (min {MIN_STD_MM}) -- the "
            f"frames carry essentially no shape information")
    if coverage < MIN_FACE_COVERAGE:
        problems.append(
            f"only {coverage*100:.0f}% of the neutral face footprint has "
            f"usable depth (min {MIN_FACE_COVERAGE*100:.0f}%)")
    if problems and not args.allow_low_quality:
        print(f"  SKIP {name}: " + "; ".join(problems)
              + "\n       (see qc/, or pass --allow-low-quality to include "
                "it anyway)")
        write_qc(out_dir / "qc" / f"{name}_REFUSED.png", npy_path, pre, rows)
        npy_path.unlink(missing_ok=True)
        return None

    info = {
        "session": name,
        "source_dir": str(rec_dir),
        "n_frames": len(rows),
        "n_unreadable": missing,
        "fps": fps,
        "duration_seconds": meta.get("duration_seconds"),
        "start_time": meta.get("start_time"),
        "frames_dropped_at_capture": meta.get("frames_dropped", 0),
        "reference_source": ref_source,
        "reference_run_frames": run_len,
        "n_reference_frames": len(ref_frames),
        "mean_valid_fraction": round(valid_frac, 4),
        "mean_face_coverage": round(coverage, 4),
        "rail_fraction": round(rail_frac, 4),
        "median_relief_mm": round(std_mm, 3),
        "label_mean_depth_mm": label_dc,
        "label_dc_spread_mm": dc_spread,
        "quality_warnings": problems,
        "labels": [r["label"] for r in rows],
        "trial_ids": [r["trial_id"] for r in rows],
        "frame_valid_frac": [round(s["valid_frac"], 4) for s in per_frame],
        "t_ms": frame_times_ms(rows),
        "device_timestamp_ms": [r["device_timestamp_ms"] for r in rows],
        "host_time_unix": [r["host_time_unix"] for r in rows],
        "preproc": pre.state_dict(),
        "label_digest": digest,
        "cache_key": cache_key,
        "array_bytes": npy_path.stat().st_size,
        "array": f"sessions/{name}.npy",
    }
    json_path.write_text(json.dumps(info), encoding="utf-8")
    write_qc(out_dir / "qc" / f"{name}.png", npy_path, pre, rows)
    notes = []
    if coverage < WARN_FACE_COVERAGE:
        notes.append(f"face coverage only {coverage*100:.0f}%")
    if dc_spread > DRIFT_WARN_MM:
        notes.append(f"label DC spread {dc_spread:.1f} mm")
    if missing:
        notes.append(f"{missing} unreadable frames")
    if problems:
        notes.append("QUALITY GATES OVERRIDDEN")
    print(f"  built {name}: {len(rows)} frames, ROI {pre.roi}, "
          f"ref {pre.reference_mm:.0f} mm, relief {std_mm:.1f} mm, "
          f"face coverage {coverage*100:.0f}%"
          + (f"   ** {'; '.join(notes)} **" if notes else ""))
    return info


def write_qc(path, npy_path, pre, rows):
    """Montage: neutral reference + evenly spaced frames, for eyeballing the
    ROI and the depth range. Silent failure here must not kill the build."""
    try:
        data = np.load(npy_path, mmap_mode="r")
        n = data.shape[0]
        idx = np.linspace(0, n - 1, min(5, n)).astype(int)
        tiles = [pre.reference_image] + [np.asarray(data[i], np.float32)
                                         for i in idx]
        names = ["reference"] + [f"#{i}" for i in idx]
        out = []
        for img, nm in zip(tiles, names):
            v = np.nan_to_num(img, nan=0.0)
            v = np.clip((v / pre.clip_mm + 1) / 2, 0, 1)
            col = cv2.applyColorMap((v * 255).astype(np.uint8),
                                    cv2.COLORMAP_JET)
            col[~np.isfinite(img)] = 0
            col = cv2.resize(col, (160, 160), interpolation=cv2.INTER_NEAREST)
            cv2.putText(col, nm, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            out.append(col)
        montage = np.hstack(out)
        top = Counter(r["label"] for r in rows).most_common(3)
        bar = np.zeros((26, montage.shape[1], 3), np.uint8)
        cv2.putText(bar, f"ROI{pre.roi}  ref={pre.reference_mm:.0f}mm  {top}",
                    (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220),
                    1, cv2.LINE_AA)
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", np.vstack([montage, bar]))
        if ok:
            path.write_bytes(buf.tobytes())
    except Exception as e:
        print(f"    (qc image failed: {e})")


def assign_splits(sessions, val_frac, test_frac, seed, previous=None):
    """Grouped split: a whole session goes to exactly one split.

    Existing sessions keep the split recorded in the previous dataset_spec.json;
    only new sessions are assigned, each to whichever split is furthest below
    its target share. This keeps proportions close to the requested fractions
    AND guarantees a session never migrates between splits as the cohort grows
    -- a plain shuffle-and-partition would move yesterday's train sessions into
    today's test set and quietly invalidate every earlier result.

    A request of 0 for val or test is honoured exactly (no split is created).
    """
    names = sorted(sessions)
    split = {}
    for n in names:
        prev = (previous or {}).get(n)
        if prev in ("train", "val", "test"):
            split[n] = prev

    targets = {"test": max(0.0, test_frac), "val": max(0.0, val_frac)}
    targets["train"] = max(0.0, 1.0 - targets["test"] - targets["val"])
    counts = Counter(split.values())
    total = len(split)
    for n in [x for x in names if x not in split]:
        total += 1
        best, best_deficit = "train", None
        for s in ("test", "val", "train"):
            if targets[s] <= 0:
                continue
            deficit = targets[s] - counts[s] / total
            if best_deficit is None or deficit > best_deficit:
                best, best_deficit = s, deficit
        split[n] = best
        counts[best] += 1
    return split


def build_clip_index(infos, split, clip_len, stride, min_purity,
                     include_unlabeled, min_clip_valid, max_gap_factor):
    """Sliding windows within a session. A window is rejected if it straddles
    a trial boundary, contains a time gap (dropped frames), has too many
    unusable frames, or is not label-pure enough."""
    clips = []
    dropped = Counter()
    for info in infos:
        name = info["session"]
        labels, trials = info["labels"], info["trial_ids"]
        vfrac = info.get("frame_valid_frac") or [1.0] * len(labels)
        t_ms = info.get("t_ms") or [None] * len(labels)
        max_gap = max_gap_factor * 1000.0 / max(1.0, float(info.get("fps", 30)))
        n = len(labels)
        for start in range(0, max(0, n - clip_len + 1), stride):
            end = start + clip_len
            if len(set(trials[start:end])) > 1:
                dropped["straddles_trial"] += 1
                continue
            ts = [t for t in t_ms[start:end] if t is not None]
            if len(ts) > 1:
                gaps = np.diff(ts)
                if gaps.size and float(np.max(gaps)) > max_gap:
                    dropped["time_gap"] += 1
                    continue
            if float(np.mean(vfrac[start:end])) < min_clip_valid:
                dropped["low_valid_depth"] += 1
                continue
            counts = Counter(labels[start:end])
            label, cnt = counts.most_common(1)[0]
            purity = cnt / clip_len
            if purity < min_purity:
                dropped["impure_label"] += 1
                continue
            if label == UNLABELED and not include_unlabeled:
                dropped["unlabeled"] += 1
                continue
            clips.append({
                "session": name, "start": start, "length": clip_len,
                "label": label, "trial_id": trials[start],
                "split": split[name], "purity": round(purity, 3),
            })
    return clips, dropped


def main():
    p = argparse.ArgumentParser(
        description="Build a training-ready dataset from D405 recordings")
    p.add_argument("--recordings", default=str(HERE / "recordings"))
    p.add_argument("--out", default=str(HERE / "dataset"))
    p.add_argument("--clip-len", type=int, default=16,
                   help="frames per training sample (16 @30fps = 0.53 s)")
    p.add_argument("--stride", type=int, default=4,
                   help="hop between clip starts, in frames")
    p.add_argument("--min-purity", type=float, default=0.9,
                   help="min fraction of the window holding the clip's label")
    p.add_argument("--min-clip-valid", type=float, default=0.4,
                   help="min mean valid-depth fraction for a clip")
    p.add_argument("--max-gap-factor", type=float, default=2.5,
                   help="reject a clip whose largest inter-frame gap exceeds "
                        "this many nominal frame intervals")
    p.add_argument("--include-unlabeled", action="store_true",
                   help="also index unlabeled clips (for pretraining)")
    p.add_argument("--ref-label", default="neutral",
                   help="label whose contiguous run defines the reference")
    p.add_argument("--ref-seconds", type=float, default=1.0)
    p.add_argument("--min-ref-frames", type=int, default=20,
                   help="minimum contiguous reference frames required")
    p.add_argument("--allow-session-start-reference", action="store_true",
                   help="if no neutral run exists, use the first frames anyway")
    p.add_argument("--allow-low-quality", action="store_true",
                   help="include sessions that fail the saturation/relief gates")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--reshuffle-splits", action="store_true",
                   help="reassign ALL sessions to splits, ignoring the previous "
                        "build (invalidates comparisons with earlier results)")
    p.add_argument("--per-frame-dc", choices=PER_FRAME_DC_CHOICES, default=None,
                   help="'roi_median' removes each frame's own DC offset, "
                        "killing slow-drift shortcuts (default: none)")
    p.add_argument("--roi", default=None, metavar="X,Y,SIDE",
                   help="explicit square ROI in raw pixels, skipping automatic "
                        "face finding. Use this when the face cannot be "
                        "separated automatically (e.g. a mouse, whose head and "
                        "body are contiguous in depth). The rig is head-fixed, "
                        "so a hand-measured box is fully reproducible.")
    numeric_overrides = {
        "out_size": "--out-size",
        "clip_mm": "--clip-mm",
        "deform_clip_mm": "--deform-clip-mm",
        "expected_distance_m": "--expected-distance",
        "search_band_m": "--search-band",
        "ref_tolerance_m": "--ref-tolerance",
        "roi_margin": "--roi-margin",
    }
    for key, flag in numeric_overrides.items():
        unit = " (metres)" if key.endswith("_m") else ""
        p.add_argument(flag, dest=key, type=float, default=None,
                       help=f"preprocessing override{unit}, default "
                            f"{DEFAULT_PARAMS[key]}")
    args = p.parse_args()

    overrides = {}
    for key in numeric_overrides:
        v = getattr(args, key)
        if v is not None:
            overrides[key] = int(v) if key == "out_size" else float(v)
    if args.per_frame_dc:
        overrides["per_frame_dc"] = args.per_frame_dc
    if args.roi:
        try:
            overrides["roi_override"] = [int(v) for v in args.roi.split(",")]
        except ValueError:
            sys.exit(f"--roi must be X,Y,SIDE in pixels, got {args.roi!r}")
        if len(overrides["roi_override"]) != 3:
            sys.exit(f"--roi must be X,Y,SIDE in pixels, got {args.roi!r}")

    rec_root = Path(args.recordings)
    out_dir = Path(args.out)
    if not rec_root.exists():
        sys.exit(f"no recordings folder at {rec_root}")
    sessions = sorted([d for d in rec_root.iterdir()
                       if d.is_dir() and (d / "metadata.json").exists()])
    if not sessions:
        sys.exit(f"no recordings found in {rec_root} -- record something first")

    print(f"{len(sessions)} recording(s) in {rec_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    infos, refused = [], []
    for d in sessions:
        info = build_session(d, out_dir, args, overrides)
        if info:
            infos.append(info)
        else:
            refused.append(d.name)
    if not infos:
        sys.exit("\nno session passed -- nothing was built. Fix the issues "
                 "above (usually subject distance or a missing neutral "
                 "period) and re-run.")

    # Pin sessions to the split they were given in an earlier build.
    previous = {}
    spec_path = out_dir / "dataset_spec.json"
    if spec_path.exists() and not args.reshuffle_splits:
        try:
            old = json.loads(spec_path.read_text(encoding="utf-8"))
            previous = {k: v.get("split")
                        for k, v in old.get("sessions", {}).items()}
        except (OSError, json.JSONDecodeError):
            pass
    split = assign_splits([i["session"] for i in infos],
                          args.val_frac, args.test_frac, args.seed, previous)
    kept = sum(1 for n, s in split.items() if previous.get(n) == s)
    if kept:
        print(f"  ({kept} session split(s) pinned from the previous build)")
    clips, clip_drops = build_clip_index(
        infos, split, args.clip_len, args.stride, args.min_purity,
        args.include_unlabeled, args.min_clip_valid, args.max_gap_factor)

    seen = sorted({lab for i in infos for lab in i["labels"]} - {UNLABELED})
    label_map = {UNLABELED: 0, **{lab: k + 1 for k, lab in enumerate(seen)}}

    with open(out_dir / "manifest_frames.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session", "frame_index", "label", "label_id", "trial_id",
                    "split", "valid_frac", "device_timestamp_ms",
                    "host_time_unix"])
        for info in infos:
            nm = info["session"]
            vf = info.get("frame_valid_frac") or [1.0] * info["n_frames"]
            for i, lab in enumerate(info["labels"]):
                w.writerow([nm, i, lab, label_map.get(lab, 0),
                            info["trial_ids"][i], split[nm], vf[i],
                            info["device_timestamp_ms"][i],
                            info["host_time_unix"][i]])

    with open(out_dir / "manifest_clips.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "session", "start", "length", "label",
                    "label_id", "trial_id", "split", "purity"])
        for k, c in enumerate(clips):
            w.writerow([k, c["session"], c["start"], c["length"], c["label"],
                        label_map.get(c["label"], 0), c["trial_id"],
                        c["split"], c["purity"]])

    per_split = Counter(c["split"] for c in clips)
    per_label = Counter(c["label"] for c in clips)
    per_split_label = defaultdict(Counter)
    for c in clips:
        per_split_label[c["split"]][c["label"]] += 1
    frame_labels = Counter(lab for i in infos for lab in i["labels"])

    spec = {
        "preproc_version": PREPROC_VERSION,
        "preproc_params": {**DEFAULT_PARAMS, **overrides},
        "reference_policy": {"ref_label": args.ref_label,
                            "ref_seconds": args.ref_seconds,
                            "min_ref_frames": args.min_ref_frames,
                            "allow_session_start_reference":
                                args.allow_session_start_reference},
        "quality_gates": {"max_rail_frac": MAX_RAIL_FRAC,
                          "min_std_mm": MIN_STD_MM,
                          "min_face_coverage": MIN_FACE_COVERAGE,
                          "overridden": args.allow_low_quality},
        "storage": "sessions/<name>.npy float16 (N,S,S) relative depth in mm, "
                   "NaN = invalid. Use preprocessing.to_model_input() with "
                   "preprocessing.input_scale(preproc_params, mode).",
        "clip": {"length": args.clip_len, "stride": args.stride,
                 "min_purity": args.min_purity,
                 "min_clip_valid": args.min_clip_valid,
                 "max_gap_factor": args.max_gap_factor,
                 "include_unlabeled": args.include_unlabeled},
        "label_map": label_map,
        "refused_sessions": refused,
        "sessions": {i["session"]: {
            "split": split[i["session"]],
            "n_frames": i["n_frames"],
            "reference_mm": i["preproc"]["reference_mm"],
            "roi": i["preproc"]["roi"],
            "mean_valid_fraction": i["mean_valid_fraction"],
            "mean_face_coverage": i["mean_face_coverage"],
            "rail_fraction": i["rail_fraction"],
            "median_relief_mm": i["median_relief_mm"],
            "label_mean_depth_mm": i["label_mean_depth_mm"],
            "label_dc_spread_mm": i["label_dc_spread_mm"],
            "reference_source": i["reference_source"],
            "fit_notes": i["preproc"].get("fit_notes", {}),
        } for i in infos},
        "split_sessions": {s: sorted([n for n, v in split.items() if v == s])
                           for s in ("train", "val", "test")},
        "counts": {
            "sessions": len(infos),
            "frames": sum(i["n_frames"] for i in infos),
            "frames_per_label": dict(frame_labels),
            "clips": len(clips),
            "clips_per_split": dict(per_split),
            "clips_per_label": dict(per_label),
            "clips_per_split_label": {k: dict(v)
                                      for k, v in per_split_label.items()},
            "clips_rejected": dict(clip_drops),
        },
    }
    (out_dir / "dataset_spec.json").write_text(json.dumps(spec, indent=2),
                                               encoding="utf-8")

    print(f"\ndataset written to {out_dir}")
    print(f"  {spec['counts']['frames']} frames, {len(clips)} clips "
          f"of {args.clip_len} frames")
    print(f"  clips per split: {dict(per_split)}")
    print(f"  clips per label: {dict(per_label)}")
    if clip_drops:
        print(f"  windows rejected: {dict(clip_drops)}")

    warnings_out = []
    if refused:
        warnings_out.append(f"{len(refused)} recording(s) were EXCLUDED: "
                            f"{refused}. See the SKIP reasons above.")
    if len(infos) < 3:
        warnings_out.append(
            f"only {len(infos)} session(s): splits are grouped by session, so "
            f"val/test are empty or tiny. Any accuracy you measure now is not "
            f"meaningful -- record several sessions per condition.")
    for s, frac in (("val", args.val_frac), ("test", args.test_frac)):
        if frac > 0 and not per_split.get(s):
            warnings_out.append(f"{s} split is empty -- record more sessions.")
    if per_split and per_split.get("train", 0) < 0.4 * len(clips):
        warnings_out.append(
            f"only {per_split.get('train', 0)}/{len(clips)} clips are in train. "
            f"With few sessions the split proportions are coarse (whole "
            f"sessions); this evens out as you add recordings.")
    labelled = {k: v for k, v in per_label.items() if k != UNLABELED}
    if len(labelled) < 2:
        warnings_out.append(
            f"only {len(labelled)} labelled class(es) present -- a classifier "
            f"needs at least 2. Tag frames with the 1-9 keys while recording.")
    for s in ("train", "val", "test"):
        labs = {k for k in per_split_label[s] if k != UNLABELED}
        if per_split.get(s) and len(labs) < len(labelled):
            warnings_out.append(f"{s} split is missing class(es): "
                                f"{sorted(set(labelled) - labs)}")
    drift = [(i["session"], i["label_dc_spread_mm"]) for i in infos
             if i["label_dc_spread_mm"] > DRIFT_WARN_MM]
    if drift:
        warnings_out.append(
            f"per-label mean depth differs by >{DRIFT_WARN_MM} mm in {drift}. "
            f"Labels are contiguous time blocks, so slow depth drift can be a "
            f"shortcut the model learns instead of facial motion. Consider "
            f"--per-frame-dc roi_median, or interleaving conditions.")
    low = [i["session"] for i in infos
           if i["mean_face_coverage"] < WARN_FACE_COVERAGE]
    if low:
        warnings_out.append(f"parts of the face were not measured in {low} "
                            f"-- check the qc/ images")
    if warnings_out:
        print("\nWARNINGS")
        for w in warnings_out:
            print(f"  ! {w}")
    print(f"\ncheck qc/*.png to confirm the face ROI is right, then:\n"
          f"  python train_baseline.py --dataset {out_dir}")


if __name__ == "__main__":
    main()
