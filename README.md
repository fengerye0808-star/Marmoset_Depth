# D405 Face Depth Recorder + Expression Pipeline

Records **depth-only** video of a head-fixed subject's face with an Intel
RealSense **D405**, in a form that trains a facial-expression model and then
runs that model on the live stream. The D405's ideal range is 7–50 cm, so a
face at ~15 cm is well inside spec.

```
record  ->  build dataset  ->  train  ->  run live
d405_recorder.py   build_dataset.py   train_baseline.py   live_infer.py
```

The three stages share one preprocessing definition (`preprocessing.py`), so
what the model sees in real time is exactly what it was trained on.
`test_pipeline.py` asserts that.

## Setup

```powershell
py -m pip install -r requirements.txt
py d405_recorder.py
```

Add PyTorch only when you get to training (CPU build is fine to start):

```powershell
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

- Plug the D405 into a **USB 3** port.
- Close the Intel RealSense Viewer first — only one process can own the camera.
- On Windows the interpreter is `py`, not `python` (the latter is a broken Store
  shortcut on the lab machines). `py --version` should print 3.11 or newer.
- No camera yet? `py d405_recorder.py --mock` runs the whole GUI on a synthetic
  camera. Note that mock sessions are written into the same `recordings/` folder
  as real ones and are only identifiable by `"device": "Mock D405"` in
  `metadata.json` — delete them so they don't reach a dataset.

---

## 1. Record

Two live panels: the normal image stream on the left, the colorized depth
stream on the right. The depth panel has a center crosshair with a live **mm
readout** — use it to set the subject at your target distance (~150 mm).

| Control | Action |
|---|---|
| **Start Recording** / **Stop** buttons | begin / end a recording |
| `Space` | toggle recording |
| `R` / `S` | start / stop |
| `1`–`9`, `0` | set the active label (tags every frame from now on) |
| `T` | trial start / end marker |
| `Q` / `Esc` | quit |

**Edit [labels.txt](labels.txt) for your experiment before recording.** One
label per line; line *n* is hotkey *n*. The active label is written into every
frame as it is captured, so labelling happens during the session rather than in
a painful post-hoc pass.

Three protocol rules make the data trainable — the pipeline depends on all
three, and enforces the first:

1. **Record a solid block of `neutral`** — ≥20 contiguous frames, best placed at
   the very start while the subject settles. `build_dataset.py` fits the face ROI
   and the session reference depth from the **longest** contiguous `neutral` run
   in the session, and uses it as the neutral face for `deform` mode. Keep the
   opening hold longer than your between-trial returns to neutral, so the
   baseline comes from a genuinely resting period rather than a post-stimulus
   one. A session with no such run is **refused**, because a reference built from
   an arbitrary expression offsets that whole session's deform space.
2. **Position the face at ~15 cm.** The ROI/reference search looks in a 6 cm
   shell around that distance. If your rig differs, change
   `expected_distance_m` in `preprocessing.py` **once** rather than passing
   `--expected-distance` per run — the value is part of the dataset cache key, so
   forgetting the flag on a later build reprocesses every session at the wrong
   distance and they all fail the fit.
3. **Record several sessions per condition, and avoid running all trials of one
   condition in one block.** Splits are grouped by session, so a single session
   gives no honest accuracy. Blocked conditions also let slow depth drift become
   a shortcut — the builder measures this and warns.

Each recording becomes `recordings/<start time>_dur<duration>s/` containing
16-bit depth PNGs, per-frame timestamps + labels, an event log, and metadata.
See [DATA_FORMAT.md](DATA_FORMAT.md) for the exact schema.

## 2. Build the training dataset

```powershell
py build_dataset.py                    # incremental; safe to re-run daily
py build_dataset.py --clip-len 24 --stride 6
```

Applies the canonical preprocessing to every frame and writes
`dataset/`: one `float16` array per session (relative depth in mm, `NaN` =
invalid), a frame manifest, a **clip** manifest (the actual training samples),
session-grouped splits, and `qc/*.png` previews.

Sessions that would corrupt training are **refused, not silently included** —
saturated depth, no measurable facial relief, a reference that could not be
fitted at the expected distance, or most of the face unmeasured. Each refusal
prints why, and writes `qc/<session>_REFUSED.png` so you can see what the
camera actually saw. `--allow-low-quality` overrides the gates if you disagree.

**Still look at `dataset/qc/*.png`.** The gates catch the failures that are
measurable; a glance catches the rest. The builder also warns about too few
sessions, empty splits, missing classes, per-label depth drift, and unreadable
frames.

Re-running only processes new recordings; editing labels in
`frame_timestamps.csv` triggers a cheap metadata-only refresh. Preprocessing
changes require `--rebuild`.

Useful knobs:

```powershell
py build_dataset.py --expected-distance 0.18    # your rig's face distance
py build_dataset.py --per-frame-dc roi_median   # kill depth-drift shortcuts
py build_dataset.py --include-unlabeled         # index clips for pretraining
```

## 3. Train

```powershell
py train_baseline.py                   # mode=deform by default
py train_baseline.py --mode absolute --epochs 40
```

A small 3D CNN over `(depth, mask) × time`. It exists to prove the pipeline and
give you a number to beat — not to be the final model. It reports per-class
metrics and a confusion matrix on the held-out **sessions**, and saves
`models/expression_model.pt` with the label map and the full preprocessing
contract baked in.

`--mode deform` subtracts the session's neutral face, leaving pure deformation.
That is usually the stronger and more subject-invariant signal; `absolute`
keeps static face shape too. Both come from the same built dataset, so trying
each costs nothing.

## 4. Run live

```powershell
py live_infer.py
```

Warm-up first: hold the subject still and neutral for ~1.5 s while it fits the
ROI and reference depth (the live equivalent of what the builder does offline).
Then it shows color | depth | the actual network input | live class
probabilities. `W` re-runs the warm-up if the rig moves, `Space` pauses, `Q`
quits. `--log predictions.csv` records every prediction with timestamps for
alignment with your other recordings.

## Exporting 3D face geometry

The depth PNGs are **not pictures** — every pixel is a distance, and the
recorder saves the camera intrinsics in each session's `metadata.json`. Together
those make each frame a metric 3D surface (~400,000 points at 30 fps). A depth
map plus intrinsics is simply the compact, lossless way to store an organised
point cloud. [export_3d.py](export_3d.py) makes the geometry explicit:

```powershell
py export_3d.py --label neutral --average --mesh --out face.ply --preview face.png
py export_3d.py --frame 120 --points --out frame120.ply
py export_3d.py --label condition_1 --sequence --every 3 --out-dir cond1_4d/
```

| Flag | What it gives you |
|---|---|
| `--average` | Per-pixel temporal median of the selected frames. Legitimate *because the head is fixed*, and it cuts depth noise by about √N — the difference between a grainy scan and a clean model. |
| `--mesh` / `--points` | Triangulated surface (with normals) or a bare point cloud. |
| `--label neutral` | Restrict to one label, e.g. a canonical neutral face; use another label for that expression's shape. |
| `--sequence` | One file per frame, i.e. 4D — shape over time. Import into Blender or MeshLab for playback. |
| `--preview face.png` | Shaded relief image, so you can check the shape without opening a 3D viewer. |
| `--roi X,Y,SIDE` | Explicit crop, same meaning as in `build_dataset.py`. |

Output is PLY (binary by default) or OBJ with `--format obj`, in **millimetres**
(`--units m` for metres), using the camera's axes: +X right, +Y down, +Z away
from the lens. Both formats open in MeshLab, CloudCompare, Blender and Open3D.
Files land in `exports/` next to the program, and the absolute path is printed.
`read_ply()` in the same module loads an export back into numpy with no extra
dependencies.

### Every frame is a separate 3D shape

A single export is **one moment**, not the recording. `--frame 200` writes the
shape at frame 200; `--average` writes one shape fused from a short window. The
filename says which (`..._frame000200_mesh.ply`, `..._avg30from000000_mesh.ply`).

To get the whole recording as 3D, use `--sequence`, which writes one file per
frame. Measured on an 18 s / 540-frame recording:

| What | Per frame | All 540 frames |
|---|---|---|
| `--sequence --mesh` | 4.8 MB | ~2.5 GB |
| `--sequence --points --decimate 2` | 0.35 MB | ~190 MB |
| the original depth PNGs | 0.29 MB | 159 MB |

Note the last row. **The depth PNGs are already the compact 4D recording** — one
3D shape per frame, at a third the size of even the lightweight export. Exporting
every frame re-encodes the same geometry roughly 15× larger, so only do it when
another program has to read the files (Blender, MeshLab, CloudCompare).

For analysis in Python, don't export at all — iterate the frames directly:

```python
from export_3d import iter_points

for idx, label, pts in iter_points("recordings/2026-07-28_15-03-28_dur18.1s",
                                   expected_distance_m=0.32, every=1):
    xyz = pts[np.isfinite(pts).all(-1)]   # (N, 3) metric points, in mm
    # pts stays an organised (H, W, 3) grid, so neighbours are still neighbours
```

That runs at about 34 ms/frame (a 540-frame recording in ~18 s) and writes
nothing to disk.

By default the export keeps only the largest connected surface inside the face
crop and a depth band around the fitted face. Without that, a lickspout or
headbar sitting near the face ends up welded into your "face" model. Use
`--keep-all` if you actually want everything in view.

Mock and synthetic recordings carry no real intrinsics; pass
`--intrinsics FX,FY,PPX,PPY` for those (`make_test_recordings.py` writes
plausible ones, so its output works unchanged).

### 3D geometry and the training data are two views of the same thing

Nothing needs re-recording to get either. The training arrays
(128×128 relative depth) are a height map on a fixed grid, which is the right
form for a network; the PLY export is the same surface in metric camera
coordinates, which is the right form for measurement and visualisation. If you
want the model to predict 3D shape rather than a class label, the stored arrays
already *are* dense per-pixel targets — only the network head and loss change.

## Testing without hardware

```powershell
py make_test_recordings.py --sessions 8     # synthetic labelled sessions
py build_dataset.py --recordings test_recordings --out test_dataset
py train_baseline.py --dataset test_dataset --epochs 25
py test_pipeline.py --full                  # 20 self-checks incl. parity
```

Each synthetic session uses a different camera distance, so a model that
generalizes across them confirms the reference normalization works. The
synthetic scenes also include the clutter a real head-fixed rig unavoidably puts
in view — torso at 20–35 cm, headbar at 12 cm, lickspout at 9.5 cm — because a
clean single-blob scene would not exercise the ROI logic at all.

`test_pipeline.py` is worth re-running after any change to `preprocessing.py`.
Beyond the parity check it asserts the failures that are otherwise invisible:
that rig clutter does not capture the ROI or the reference, that clipping happens
after the deform difference, that reference holes are masked rather than faked,
and that session splits never migrate as the cohort grows.

---

## Scaling up to a real model

The pipeline is the part that is hard to change later; the model is not. When
you have real data:

- **Get more sessions, not longer ones.** Generalization across sessions and
  subjects is what matters, and splits are grouped by session.
- **Use the unlabeled frames.** `build_dataset.py --include-unlabeled` indexes
  them; pretrain (masked autoencoding or contrastive) and fine-tune on the
  labelled subset. Depth of a head-fixed face is highly structured, so
  self-supervision pays off quickly.
- **Swap the model.** Anything consuming `(2, T, 128, 128)` drops in — R(2+1)D,
  a video transformer, or per-frame CNN + temporal head. Only
  `DepthExpressionNet` in `train_baseline.py` needs to change.
- **Check the temporal window.** `--clip-len` is the expression timescale you
  assume. Sweep it; slow states want longer windows than fast twitches.
- **Consider landmark/action-unit supervision** if you need interpretable
  outputs rather than one class per clip — the stored arrays support dense
  targets, only the head and loss change.

If you change anything in `preprocessing.py`, bump `PREPROC_VERSION`, rebuild
with `--rebuild`, and retrain. Old checkpoints are then refused instead of
silently degrading.

## Files

| File | Role |
|---|---|
| [d405_recorder.py](d405_recorder.py) | recording GUI, live labelling |
| [labels.txt](labels.txt) | your experiment's label vocabulary |
| [preprocessing.py](preprocessing.py) | **the shared contract**: raw depth → model input |
| [build_dataset.py](build_dataset.py) | recordings → training arrays + manifests |
| [depth_dataset.py](depth_dataset.py) | PyTorch Dataset, augmentation, class weights |
| [train_baseline.py](train_baseline.py) | 3D CNN baseline + honest evaluation |
| [live_infer.py](live_infer.py) | real-time expression readout |
| [export_3d.py](export_3d.py) | depth → 3D point clouds / meshes (PLY, OBJ) |
| [make_test_recordings.py](make_test_recordings.py) | synthetic data for testing |
| [test_pipeline.py](test_pipeline.py) | self-checks, incl. train/deploy parity |
| [DATA_FORMAT.md](DATA_FORMAT.md) | exact on-disk schemas |

## Tuning

`d405_recorder.py`: `WIDTH/HEIGHT/FPS` (stream profile),
`DEPTH_MIN_M`/`DEPTH_MAX_M` (preview color range only), `PNG_COMPRESSION`
(0 = fast/large … 9 = slow/small).

`preprocessing.py` `DEFAULT_PARAMS`: `expected_distance_m` / `search_band_m`
(where to look for the face), `ref_tolerance_m` and `min_relief_mm` (when to
refuse a fit), `out_size` (network input size), `clip_mm` (relative-depth range
kept), `deform_clip_mm` (deform-mode scaling), `roi_margin`. Override per build,
e.g. `--expected-distance 0.18 --clip-mm 60 --out-size 96`.

If you change any of them, the dataset must be rebuilt (`--rebuild`) — the
values are recorded in `dataset_spec.json` and in every checkpoint, and a
mismatch is a hard error rather than a silent accuracy loss.

### Loading raw depth yourself

```python
import cv2, json, numpy as np
from pathlib import Path

rec = Path("recordings/2026-07-28_14-30-05_dur62.4s")
meta = json.loads((rec / "metadata.json").read_text())
scale = meta["camera"]["depth_scale_m_per_unit"]

for p in sorted((rec / "depth").glob("*.png")):
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    depth_m = np.where(raw > 0, raw * scale, np.nan)   # 0 = no measurement
```
