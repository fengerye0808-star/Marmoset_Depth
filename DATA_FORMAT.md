# Data format

Two layers, deliberately separated:

- **Raw recordings** (`recordings/`) — lossless, never overwritten, the archive
  of record. Depth exactly as the camera reported it.
- **Built dataset** (`dataset/`) — derived, disposable, regenerable at any time
  with `build_dataset.py`. Preprocessed and indexed for training.

If you change your mind about preprocessing, you throw away `dataset/` and
rebuild. You never touch `recordings/`.

---

## 1. Raw recording (written by `d405_recorder.py`)

```
recordings/
  recordings_log.csv                     one row per recording
  2026-07-28_14-30-05_dur62.4s/
    depth/depth_000000.png ...           16-bit PNG per frame
    frame_timestamps.csv                 per-frame timing + LABEL + trial
    events.csv                           label changes / trial markers
    metadata.json                        camera, duration, label vocabulary
```

Folder name is `<start time>_dur<duration>s`. `_2`, `_3` … are appended if two
recordings would otherwise collide.

### `depth/depth_NNNNNN.png`
Single-channel 16-bit PNG, full sensor resolution (848×480 by default),
losslessly compressed.

```
depth_meters = png_pixel_value * metadata.json["camera"]["depth_scale_m_per_unit"]
```

`0` means **no depth measurement** at that pixel (stereo could not match) — it
is *not* zero distance. Always mask it out. On the D405 the scale is 0.0001,
i.e. 0.1 mm per unit.

**These files are 3D data, not images.** With the intrinsics stored alongside
them, every pixel back-projects to a metric point:

```python
Z = raw * depth_scale                 # metres
X = (u - ppx) * Z / fx
Y = (v - ppy) * Z / fy
```

A depth map plus intrinsics is an *organised* point cloud — the same geometry as
a PLY file, roughly 30× smaller, and with neighbour relationships intact (which
is what makes meshing and masked resizing possible at all). That is why the
recorder stores depth this way rather than as point clouds.
[export_3d.py](export_3d.py) converts to PLY/OBJ when you want explicit
geometry. Camera axes are +X right, +Y down, +Z away from the lens.

### `frame_timestamps.csv`
| column | meaning |
|---|---|
| `frame_index` | 0-based; matches the PNG filename |
| `filename` | relative path to the depth PNG |
| `device_timestamp_ms` | camera clock — use this for inter-frame timing |
| `host_time_unix` | PC clock at arrival — use this to sync with other gear |
| `host_time_iso` | same, human-readable |
| `label` | active label when the frame was captured |
| `trial_id` | active trial number, `0` = no trial |

A row exists only for a frame that was successfully written, so the CSV is the
authoritative frame list.

### `events.csv`
`elapsed_s, frame_index, kind, value` — `kind` is `label`, `trial_start`, or
`trial_end`. Use this for onset timing; use the per-frame `label` column for
training labels.

`frame_index` here is the writer's count at the moment of the event, which can
lag the capture thread slightly under heavy disk load. For frame-accurate
onsets, prefer the per-frame `label` column (it is attached at capture time).

### `metadata.json`
`format_version`, `labels` (vocabulary used), `label_frame_counts`,
`start_time`, `end_time`, `duration_seconds`, `frames_saved`,
`frames_dropped`, `frames_failed_to_write`, `effective_fps`, and `camera`
(device, serial, firmware, stream profile, `depth_scale_m_per_unit`, full depth
intrinsics `fx/fy/ppx/ppy`).

`frames_dropped > 0` means the disk could not keep up; `frames_failed_to_write`
means writes actually failed. Both are 0 in a healthy recording.

---

## 2. Built dataset (written by `build_dataset.py`)

```
dataset/
  dataset_spec.json      preprocessing params, label map, splits, counts
  manifest_frames.csv    one row per frame
  manifest_clips.csv     one row per training sample  <-- train on this
  sessions/<name>.npy    (N, S, S) float16 relative depth in mm, NaN = invalid
  sessions/<name>.json   per-frame labels/trials/timestamps + preproc state
  qc/<name>.png          visual check of the ROI and depth range
```

### `sessions/<name>.npy`
`float16`, shape `(n_frames, out_size, out_size)`, default 128×128. Values are
**relative depth in millimetres**: the face surface minus that session's
reference distance, clipped to ±40 mm. `NaN` = no valid measurement.

Relative, not absolute, because it makes the data invariant to how far the
camera happened to sit from the subject — that is what allows pooling across
sessions and subjects. The absolute reference is kept in the session JSON
(`preproc.reference_mm`), so nothing is lost.

### `manifest_clips.csv`
A clip is a short window of frames — expressions are motion, so a single frame
is not a sample. Columns: `clip_id, session, start, length, label, label_id,
trial_id, split, purity`. Default window is 16 frames (0.53 s at 30 fps) with
stride 4. `purity` is the fraction of the window holding the clip's label;
windows below `--min-purity` (0.9) and windows straddling two trials are
dropped.

### Splits
Grouped **by session**: a session is entirely in train, val, or test. Frames
within a session are far too correlated to split randomly — doing so inflates
accuracy dramatically.

Existing sessions keep the split recorded in the previous `dataset_spec.json`;
only new sessions are assigned, each to whichever split is furthest below its
target share. So a session **never migrates** between splits as you add
recordings (which would move last week's training sessions into this week's test
set and silently invalidate every earlier result), while proportions stay close
to `--val-frac` / `--test-frac`. `--reshuffle-splits` forcibly reassigns
everything, and says so.

---

## 3. The preprocessing contract

`preprocessing.py` is the single definition, versioned by `PREPROC_VERSION`
(currently `v2`). It is imported by the dataset builder, the training dataset,
**and** live inference, so the tensor the network sees in the lab is the tensor
it was trained on. `test_pipeline.py` asserts that parity directly.

**Fitted once per session** (from the neutral period, or the live warm-up):

1. Keep only pixels within `search_band_m` (6 cm) of `expected_distance_m`
   (15 cm) — a thin shell around where you positioned the face.
2. Require a pixel to be in that band across most reference frames, then take
   the **largest connected component**. The face is one coherent surface; a
   lickspout or headbar at a similar depth is not connected to it.
3. ROI = that component's box, squared and padded, then held fixed for the
   session (valid because the rig is head-fixed).
4. `reference_mm` = median depth of **that component only**.
5. **Refuse** the fit if the reference is more than `ref_tolerance_m` (5 cm)
   from the expected distance, or if the surface is flatter than
   `min_relief_mm` (3 mm — a face is curved, a headbar is not).

Steps 1, 2, 4 and 5 exist because the obvious alternative — the bounding box of
everything in 6–40 cm — fails catastrophically on a real rig. The animal's
torso, the headbar and the spout are all in range, so the ROI grows to the whole
frame and the reference lands on the torso. The face is then >40 mm from the
reference and clips to a constant: **zero expression signal, while every summary
statistic still reports a healthy session.** `test_pipeline.py` reproduces that
scene and asserts it does not happen.

**Per frame:**

6. Crop the ROI, convert to depth relative to `reference_mm`, in millimetres.
7. Pixels outside ±`clip_mm` become **invalid (`NaN`), not clamped**. Clamping
   invents a flat surface where there is no usable measurement; `NaN` says "no
   data" and shows up honestly in the coverage QC.
8. Masked area-resize to 128×128 (invalid pixels never bleed into valid ones).

Storage is **unclipped** millimetres, because clipping before a difference would
destroy deformation.

`to_model_input` then produces the final `(2, T, H, W)` tensor: channel 0 is
depth scaled to [-1, 1] — divided by `input_scale(params, mode)`, and clipped
**after** any deform subtraction — and channel 1 is the validity mask. The mask
channel matters: without it the network cannot tell "flat surface" from "no
data". In `deform` mode a pixel missing from *either* the frame or the reference
is masked invalid, so a hole in the neutral face never leaks absolute depth into
a deformation map.

### Quality gates

`build_dataset.py` **refuses** a session (rather than including it silently) when

| gate | meaning |
|---|---|
| `rail_fraction > 2%` | pixels pinned at the ±clip limit — reference on the wrong surface |
| `median_relief_mm < 0.5` | frames carry essentially no shape information |
| `mean_face_coverage < 50%` | most of the neutral face footprint was not measured |

Coverage is measured against the **neutral face footprint**, not the whole ROI —
a square ROI around an oval face is only ~50% face even with a perfect
measurement, so a raw valid-pixel fraction is not a quality signal.
`--allow-low-quality` overrides the gates; the reason is recorded in
`dataset_spec.json`.

The builder also reports `label_mean_depth_mm` per session and warns when the
per-label means differ by more than 2 mm. Labels are contiguous time blocks, so
slow depth drift can become a shortcut the model learns instead of facial
motion. Fix it at the source with `--per-frame-dc roi_median`, which removes
each frame's own DC offset.

Two views come free from the same stored array:

- `absolute` — relative depth as stored; keeps the subject's static face shape.
- `deform` — minus the session's neutral face; **pure deformation**, and
  subject-invariant. Usually the stronger signal, and the training default.

The checkpoint records `preproc_version`, all params, the label order, and the
mode. Loading a checkpoint built under a different preprocessing version is a
hard error rather than a silent accuracy loss.

Changing preprocessing means: bump `PREPROC_VERSION`, then
`py build_dataset.py --rebuild` and retrain. Old checkpoints are then
correctly refused.
