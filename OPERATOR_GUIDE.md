# D405 Face Depth Recorder — Operator Guide

For lab members recording facial depth data on the head-fixed rig.

- **Recording a session:** read sections 1–6. No programming needed.
- **Processing data afterwards:** sections 7–10.
- **Something went wrong:** section 11.
- **Stick section 12 next to the rig.**

---

## 1. What this tool does

It records the **3D shape of the subject's face** over time with an Intel
RealSense D405 depth camera, tags every frame with the condition you are
running, and stores it so a facial-expression model can be trained from it.

It records **depth only.** The left-hand image on screen is for aiming and is
never saved.

Every recorded frame is already a complete 3D surface of the face — roughly
400,000 points, 30 times a second. You do not have to do anything special to
"get 3D"; it is what the camera measures.

---

## 2. First time on a new computer

Do this once. Skip to section 3 if the rig is already set up.

**Open a terminal in the program folder.** In File Explorer, navigate to the
folder containing `d405_recorder.py`. Click once in the address bar at the top,
type `powershell`, and press Enter. A blue window opens, already pointing at the
right folder. Every command in this guide is typed there.

**Check Python.** Type:

```
py --version
```

You should see `Python 3.11` or newer. If you instead see a message about the
Microsoft Store, stop and tell whoever maintains the rig — **do not install
Python from the Store**, it will not work.

> Use **`py`**, not `python`. On this lab's machines `python` is a broken
> Windows shortcut and will fail.

**Install the software it needs:**

```
py -m pip install -r requirements.txt
```

Wait for it to finish. Only needed once per computer. (If you also intend to
train models on this machine, see section 9.)

**Set your rig's working distance.** The software needs to know roughly how far
the face sits from the camera. The default is **150 mm**. If your rig differs,
open `preprocessing.py` in a text editor, find the line

```
    "expected_distance_m": 0.15,
```

and change `0.15` to your distance **in metres** (250 mm → `0.25`). Do this
once, and everything downstream stays consistent. Write the number in the lab
notes.

---

## 3. Before each session

1. Plug the D405 into a **USB 3** port (blue, or marked SS).
2. Close the **Intel RealSense Viewer** and any other camera program. Only one
   program can use the camera at a time.
3. Check **free disk space**. Recording uses about **0.5 GB per minute**, so
   ~30 GB per hour. Nothing warns you when the disk fills, and a full disk
   loses frames. Budget generously before a long session.
4. Set your condition names — see section 4.
5. Start the program:

```
py d405_recorder.py
```

---

## 4. Setting up your condition labels

Open `labels.txt` in a text editor and list your conditions, one per line.
Lines starting with `#` are comments and are ignored.

```
unlabeled
neutral
sucrose
quinine
airpuff
```

Rules that matter:

- **`neutral` must be spelled exactly like that, all lower case.** `Neutral` or
  `NEUTRAL` will not be recognised and **every session will be rejected later**,
  after the animal has gone.
- **Keep `unlabeled` as the first line.** If you leave it out it gets inserted
  automatically, which produces two confusingly identical buttons.
- **Maximum 10 labels.** Only the first ten get a button and a key. Anything
  beyond appears as a grey "(+N more, no hotkey)" note and can never be
  selected. So: `unlabeled`, `neutral`, and up to 8 conditions.
- **Restart the program after editing.** It reads the file once at startup.
- If you edit in Notepad, make sure it saves as `labels.txt` and not
  `labels.txt.txt`. If the file cannot be read the program silently falls back
  to only `unlabeled` and `neutral`, and none of your conditions will exist.

---

## 5. Recording procedure

Steps 3 and 6 are the ones that decide whether the session is usable.

1. **Position the subject.** Watch the number at the crosshair in the depth
   panel and bring it to your rig's working distance (150 mm by default).
   *The crosshair measures the exact centre of the image, whatever is there.* If
   the face is off-centre it may be reading the headbar, the spout, or the
   torso. If it shows `center: --` there is no depth at that point at all.
2. **Click on the recorder window** so it is in front. Keyboard keys only work
   while it has focus — if you click the terminal or another window, Space and
   the number keys do nothing, with no warning.
3. **Press `2`** (`neutral`). Confirm the button highlights blue and the label
   appears on the video panel.
4. **Press `Space`** to start recording.
5. **Hold the subject still and neutral for 3–4 seconds.** No stimulus yet.
   This is the resting baseline for the whole session.
6. **Press `T`** to open a trial.
7. **Press your condition key** (`3`, `4`, `5` …) **at the moment the stimulus
   starts.** The label applies from that frame onward, so your key press *is*
   the event time.
8. **Hold that condition for at least 2 seconds.** Blocks shorter than about
   0.6 s produce **no training data at all** (see section 8).
9. **Press `T`** to close the trial, then **press `2`** to return to neutral.
10. Repeat 6–9 for each trial, **mixing up the order of conditions.** Do not run
    all of A then all of B (section 10).
11. **Press `S`** to stop, and read the save message (section 6).

Two more things:

- **Record 5 or more separate sessions**, not one long one. Press Stop, then
  Start again. Section 9 explains why this is not optional.
- **Keep the opening neutral hold the longest neutral stretch in the session.**
  The software picks the *longest* run of `neutral` as the baseline, so keep your
  between-trial returns to neutral shorter than the opening hold (a couple of
  seconds is plenty).

---

## 6. The window, and the save message

Two live panels on top, controls below.

**Top left — aiming image.** Usually colour, but the camera may fall back to
infrared or to nothing; the caption above the panel says which
("Color stream", "Infrared stream", "No image stream"). Shows the active label,
trial number, and a blinking red dot while recording. **Never saved.**

**Top right — depth image.** This is what gets recorded. Colour = distance. The
centre crosshair shows distance in millimetres.

| Control | Key | What it does |
|---|---|---|
| ● Start Recording | `Space` or `R` | Starts a recording |
| ■ Stop | `Space` or `S` | Stops and saves |
| REC timer | — | Elapsed time and frames written |
| `1 unlabeled` | `1` | "Recorded but excluded from training." Use for setup and transitions. |
| `2 neutral` | `2` | The resting face. **Required.** Also use between trials. |
| `3 …` onward | `3`–`9`, `0` | Your conditions, in `labels.txt` order |
| `T trial start` | `T` | Opens a trial; press again to close |

Three behaviours to know:

- **The selected label persists across recordings.** It is not reset when you
  press Stop. If a condition is still selected when you start the next
  recording, that recording begins tagged with it. **Always press `2` before
  `Space`.**
- **Pressing a number with no matching label does nothing at all** — no beep, no
  message. With five labels, keys `6`–`0` are silent.
- **If the frame count stops rising while the timer keeps going**, the camera has
  stopped delivering. Look for grey `CAMERA ERROR` text at the bottom of the
  window. Stop, fix the USB connection, and start again.

### Reading the save message

A healthy session prints exactly this and nothing more:

```
Saved 578 frames (19.3 s) → 2026-07-29_10-46-53_dur19.3s
```

**There is no "0 dropped frames" message — good news is silence.** Report any of
these to whoever maintains the rig:

| In the message | Meaning |
|---|---|
| `[WARNING: N frames dropped]` | Disk could not keep up; those frames are gone |
| `[WARNING: N frames failed to write]` | Writes failed — disk full, permissions, or a disconnected drive |
| `[WRITER ERROR: …]` | The saving process crashed mid-session |
| `[folder kept as …]` | The folder could not be renamed; data is fine |

`Q` or `Esc` quits, and waits up to 90 seconds for saving to finish. If you kill
the program before it finishes, you can be left with a folder of images and no
`metadata.json`, which later tools skip entirely.

---

## 7. Where the data is

Everything goes to the **`recordings`** folder inside the program folder. This
path is fixed — you cannot record straight to a network or external drive; copy
it afterwards.

```
recordings/
  2026-07-29_10-46-53_dur19.3s/
      depth/                    one 16-bit PNG per frame — the 3D data
      frame_timestamps.csv      each frame's time, label and trial number
      events.csv                label changes and trial markers
      metadata.json             camera settings, duration, frame counts
  recordings_log.csv            one line per recording ever made
```

Nothing is overwritten. **Back up this whole folder** — it is the original data,
and everything else can be rebuilt from it.

The PNGs are **not photographs.** Each pixel holds a distance, not a brightness.
Opened in an image viewer they look like odd grey pictures. That is normal.

> **Mock recordings land here too.** If anyone runs the program with `--mock` to
> demo the interface, those sessions are written into `recordings/` looking
> exactly like real ones. The only marker is `"device": "Mock D405"` inside
> `metadata.json`. Delete them, or they may end up in your dataset. There are
> currently four such folders from 2026-07-28 11:17.

**Do not rename or delete session folders.** The folder name is the session's
identity; renaming one can move it between the training and testing groups.

---

## 8. Building the training dataset

Run this after a recording day:

```
py build_dataset.py
```

It reads `recordings/` and writes a `dataset/` folder. Re-running is safe —
sessions already processed are skipped.

It **rejects** sessions that would spoil training, and says why:

| Message | Meaning | What to do |
|---|---|---|
| `no contiguous run of 'neutral' >= 20 frames` | No baseline was recorded, or `neutral` was misspelled in `labels.txt` | **Recoverable:** if part of the session really was resting, edit those rows' `label` column to `neutral` in `frame_timestamps.csv` and re-run. Last resort: `py build_dataset.py --allow-session-start-reference`, which uses the opening frames whatever their label — a worse baseline. |
| `no face-sized surface found at 15 +/- 6 cm` | Nothing face-like at the expected distance | Check the distance you set in section 2. For **small subjects like mice**, whose head and body merge in depth, automatic detection cannot work at all — use an explicit box: `py build_dataset.py --roi X,Y,SIDE` (pixels, from the depth image). |
| `only NN% of the neutral face footprint has usable depth` | The face was not measured for much of the session, usually because the subject moved out of position | Look at `dataset/qc/*_REFUSED.png` to see what the camera saw |
| `fitted reference depth … is flat` | It locked onto a headbar, wall, or holder instead of a face | Use `--roi` |

After the summary it also prints a **WARNINGS** block. Those are not rejections —
the dataset was still built. The two that matter most:

- `only N labelled class(es) present` — conditions were never selected while
  recording. A classifier needs at least two.
- `per-label mean depth differs by >2.0 mm` — your conditions may be
  distinguishable by slow rig drift rather than by facial movement, which lets a
  model score well for the wrong reason. Interleave conditions better, or ask
  for `--per-frame-dc roi_median` to be used.

**Always look at `dataset/qc/*.png`** — a strip showing the resting face plus
sample frames. If it does not look like a face, the data is unusable and one
glance tells you. Note that sessions rejected very early produce no QC image at
all.

### Why label blocks must be long enough

Training samples are 16-frame windows (about half a second), and a window is only
used if at least 15 of its 16 frames carry the same label. So:

| Condition block | Training clips produced |
|---|---|
| under ~0.5 s | **none** |
| ~0.6 s | 1, maybe |
| 2 s | about 12 |

Hold each condition for **at least 2 seconds**.

### Correcting labels afterwards

You can fix the `label` column in `recordings/<session>/frame_timestamps.csv`
and re-run; the change is picked up without reprocessing the images. But:

- **Only edit the `label` column.** Never add, delete or reorder rows — the row
  order defines which frame is which.
- Open it in a **plain text editor**, not Excel. Excel rewrites timestamps and
  can corrupt the file.
- Label names must match `labels.txt` exactly, including case.

---

## 9. Training a model

Training needs PyTorch, installed once:

```
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then:

```
py train_baseline.py
```

It trains on `dataset/`, saves to `models/`, and reports accuracy on sessions it
has never seen.

**Why you need at least 5 sessions.** Whole sessions are kept separate: some for
training, some for testing, some for validation. Frames from one session are
nearly identical to each other, so testing on a session the model trained on
would give a meaninglessly high score. Sessions are assigned in the order they
were recorded — the 1st goes to training, the 2nd to testing, the 4th to
validation. **With fewer than 4 usable sessions there is no validation set, and
with only one there is no honest score at all.** Aim for several sessions per
condition, on different days.

To run a trained model live on the camera:

```
py live_infer.py
```

Hold the subject still and neutral during the warm-up, then it shows the
predicted expression in real time. `W` redoes the warm-up if the rig is moved,
`Q` quits.

---

## 10. Getting 3D face models out

```
py export_3d.py --average --mesh --label neutral
```

Writes a 3D file into the **`exports`** folder, and prints the full path. Opens
in MeshLab, CloudCompare or Blender.

**By default it uses the most recent recording.** To pick another, pass
`--recording recordings\<folder name>`.

| Option | Effect |
|---|---|
| `--recording <folder>` | Which session (default: newest) |
| `--frame 200` | The shape at one single frame |
| `--average` | Combines ~1 s into one cleaner model; needs the subject to be still |
| `--mesh` / `--points` | A joined-up surface, or just the points |
| `--label neutral` | Only frames with that label |
| `--sequence` | One file per frame — the whole recording as 3D over time |
| `--preview face.png` | Also writes a picture, to check without a 3D program |
| `--auto-distance` | Measures the subject's distance from the data instead of assuming it |

A single export is **one moment**, not the whole recording; the filename says
which frames are inside. `--sequence` gives every frame but is large — about
2.5 GB for 18 seconds. The original PNGs already contain every 3D shape at a
fraction of that, so only export what another program has to open.

---

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Python was not found` / Microsoft Store message | You typed `python`. Use `py`. |
| `ModuleNotFoundError: No module named 'cv2'` | Dependencies not installed. Run `py -m pip install -r requirements.txt` |
| `can't open file 'd405_recorder.py'` | The terminal is in the wrong folder. Reopen it via the File Explorer address-bar trick in section 2. |
| Error box: "Could not open the RealSense D405" | Camera not on USB 3, or another program (RealSense Viewer) has it. Close that program, reseat the cable, try another port. |
| Window opens but both panels are black | Camera not delivering frames. Check the grey text at the bottom of the window for `CAMERA ERROR`. |
| Space / number keys do nothing | The recorder window is not in front. Click on it first. |
| A number key does nothing specifically | There is no label on that line of `labels.txt`. |
| Crosshair shows `center: --` | No depth measurement at the image centre — very dark fur, a shiny surface, or too close. Adjust position or lighting. |
| Timer runs but frame count is frozen | Camera stopped delivering. Stop, fix USB, re-record. |
| `build_dataset.py` says "no session passed" | Read the SKIP reason printed for each session; the table in section 8 covers each one. |
| Recording started already tagged with a condition | The label persisted from the previous recording. Press `2` before `Space` next time. |

If in doubt, **keep the data and ask** — almost nothing is unrecoverable as long
as the `recordings` folder is intact.

---

## 12. Quick reference — stick this on the rig

```
  BEFORE    camera on USB 3      RealSense Viewer closed
            disk space: ~0.5 GB per minute of recording
            py d405_recorder.py

  RECORD    click the recorder window first (keys need focus)
            2          select neutral
            Space      start
            hold still and neutral 3-4 s      <- the baseline
            T          open trial
            3/4/5      condition, AT stimulus onset
            hold 2 s or more                  <- shorter = no training data
            T          close trial
            2          back to neutral
            ...        repeat, mixing the order of conditions
            S          stop

  KEYS      Space start/stop   R start   S stop
            1-9,0  select label      T  trial marker
            Q/Esc  quit

  CHECK     save message reads "Saved N frames (T s) -> folder"
            and contains NO [WARNING] or [WRITER ERROR]

  DATA      recordings\<date>_<time>_dur<length>s\
            back it up; do not rename folders

  GOAL      5+ separate sessions, conditions interleaved
```

Questions about the rig or the data go to whoever maintains this tool.
