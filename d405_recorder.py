#!/usr/bin/env python3
"""
D405 Face Depth Recorder
========================

Records depth video of a head-fixed subject's face with an Intel RealSense
D405 short-range stereo depth camera (ideal working range 7-50 cm, so a
~15 cm face distance is well inside spec).

Live preview shows two panels side by side:
  left  - the normal image stream (RGB from the D405's left imager)
  right - the colorized depth stream, with a crosshair readout of the
          distance at the image center (useful for positioning at ~15 cm)

Only DEPTH data is recorded. Each recording becomes one sub-folder inside
"recordings/" (created next to this script), named with the start time and,
once finished, the duration, e.g.

    recordings/2026-07-28_14-30-05_dur12.3s/
        depth/depth_000000.png ...   16-bit PNGs, value * depth_scale = meters
        frame_timestamps.csv         per-frame device + host timestamps
        metadata.json                camera intrinsics, depth scale, duration...

Controls
    [Start Recording] / [Stop] buttons
    Space  toggle recording          R  start        S  stop
    Q / Esc quit

Run without a camera attached for a synthetic test pattern:
    python d405_recorder.py --mock

Dependencies: pyrealsense2, opencv-python, numpy  (see requirements.txt)
"""

import argparse
import collections
import csv
import datetime as dt
import json
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

# ---------------------------- configuration ----------------------------
WIDTH, HEIGHT, FPS = 848, 480, 30      # D405 stream profile
DEPTH_MIN_M = 0.05                     # display color-range near limit
DEPTH_MAX_M = 0.35                     # display color-range far limit
DISP_W, DISP_H = 424, 240              # on-screen size of each panel
PNG_COMPRESSION = 1                    # 0=fast/large ... 9=slow/small
QUEUE_SIZE = 240                       # writer backlog before dropping frames
HERE = Path(__file__).resolve().parent
RECORD_ROOT = HERE / "recordings"
LABELS_FILE = HERE / "labels.txt"
FORMAT_VERSION = 2                     # recording layout version

BG = "#151820"
FG = "#e8e8e8"
DIM = "#9aa0ad"
RED = "#e5484d"
GREEN = "#46a758"

Frame = collections.namedtuple(
    "Frame", ["color", "depth", "depth_vis", "ts_ms", "frame_no", "host_time"]
)


def now_iso():
    return dt.datetime.now().isoformat(timespec="milliseconds")


def load_labels():
    """Read the experiment's label vocabulary from labels.txt.
    Falls back to a minimal vocabulary if the file is missing."""
    try:
        lines = LABELS_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ["unlabeled", "neutral"]
    labels, seen = [], set()
    for line in lines:
        name = line.split("#", 1)[0].strip()
        if name and name not in seen:
            seen.add(name)
            labels.append(name)
    if not labels:
        return ["unlabeled", "neutral"]
    if labels[0] != "unlabeled":
        labels.insert(0, "unlabeled")
    return labels


# ---------------------------- camera sources ----------------------------
class RealSenseSource:
    """Live D405 via librealsense. Falls back to IR / depth-only preview
    if the color stream profile is not available."""

    def __init__(self):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is not installed. Run: pip install pyrealsense2\n"
                "(or use --mock to test the GUI without a camera)"
            )
        self.pipeline = rs.pipeline()
        self.left_kind = "color"  # what the left preview panel shows

        attempts = [
            ("color", lambda c: c.enable_stream(rs.stream.color, WIDTH, HEIGHT,
                                                rs.format.bgr8, FPS)),
            ("infrared", lambda c: c.enable_stream(rs.stream.infrared, 1, WIDTH,
                                                   HEIGHT, rs.format.y8, FPS)),
            ("none", lambda c: None),
        ]
        last_err = None
        for kind, add_second in attempts:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
            add_second(cfg)
            try:
                self.profile = self.pipeline.start(cfg)
                self.left_kind = kind
                break
            except RuntimeError as e:
                last_err = e
        else:
            raise RuntimeError(
                "Could not start the RealSense pipeline "
                f"({WIDTH}x{HEIGHT}@{FPS}). Is the D405 plugged into USB 3?\n"
                f"Last error: {last_err}"
            )

        dev = self.profile.get_device()
        self.depth_sensor = dev.first_depth_sensor()
        self.depth_scale = float(self.depth_sensor.get_depth_scale())

        def info(key):
            try:
                return dev.get_info(key)
            except RuntimeError:
                return "unknown"

        self.device_name = info(rs.camera_info.name)
        self.serial = info(rs.camera_info.serial_number)
        self.firmware = info(rs.camera_info.firmware_version)

        intr = (self.profile.get_stream(rs.stream.depth)
                .as_video_stream_profile().get_intrinsics())
        self.intrinsics = {
            "width": intr.width, "height": intr.height,
            "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
            "model": str(intr.model), "coeffs": list(intr.coeffs),
        }

        # Fixed color range (no histogram equalization) so the depth image
        # of a face at ~15 cm looks stable frame to frame.
        self.colorizer = rs.colorizer()
        try:
            self.colorizer.set_option(rs.option.histogram_equalization_enabled, 0)
            self.colorizer.set_option(rs.option.min_distance, DEPTH_MIN_M)
            self.colorizer.set_option(rs.option.max_distance, DEPTH_MAX_M)
        except RuntimeError:
            pass  # keep librealsense defaults if an option is missing

    def read(self):
        frames = self.pipeline.wait_for_frames(5000)
        depth = frames.get_depth_frame()
        if not depth:
            return None
        host_time = time.time()
        depth_img = np.asanyarray(depth.get_data()).copy()
        vis = np.asanyarray(self.colorizer.colorize(depth).get_data())
        depth_vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        color_img = None
        if self.left_kind == "color":
            c = frames.get_color_frame()
            if c:
                color_img = np.asanyarray(c.get_data()).copy()
        elif self.left_kind == "infrared":
            ir = frames.get_infrared_frame(1)
            if ir:
                color_img = cv2.cvtColor(np.asanyarray(ir.get_data()),
                                         cv2.COLOR_GRAY2BGR)
        return Frame(color_img, depth_img, depth_vis,
                     float(depth.get_timestamp()),
                     int(depth.get_frame_number()), host_time)

    def metadata(self):
        return {
            "device": self.device_name,
            "serial_number": self.serial,
            "firmware_version": self.firmware,
            "stream": {"width": WIDTH, "height": HEIGHT, "fps": FPS,
                       "depth_format": "Z16"},
            "depth_scale_m_per_unit": self.depth_scale,
            "depth_intrinsics": self.intrinsics,
        }

    def stop(self):
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass


class MockSource:
    """Synthetic face-like depth bump for testing the GUI without hardware."""

    def __init__(self):
        self.depth_scale = 0.0001  # 100 um/unit, same as the D405 default
        self.left_kind = "color"
        self.device_name = "Mock D405"
        self.serial = "0000"
        self.firmware = "-"
        self.frame_no = 0
        self.t0 = time.time()
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
        self._r2 = (((xx - WIDTH / 2) / 140.0) ** 2
                    + ((yy - HEIGHT / 2) / 180.0) ** 2)

    def read(self):
        time.sleep(1.0 / FPS)
        t = time.time() - self.t0
        base_m = 0.15 + 0.008 * np.sin(t * 1.5)          # face at ~15 cm
        bump = np.exp(-self._r2) * (0.03 + 0.01 * np.sin(t * 4.0))
        depth_m = np.where(self._r2 < 2.5, base_m - bump, 0.0)
        depth = (depth_m / self.depth_scale).astype(np.uint16)

        clipped = np.clip((depth_m - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M),
                          0, 1)
        vis8 = (255 - clipped * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(vis8, cv2.COLORMAP_JET)
        depth_vis[depth == 0] = 0

        shade = (np.exp(-self._r2) * 180 + 40).astype(np.uint8)
        color = cv2.cvtColor(shade, cv2.COLOR_GRAY2BGR)
        cv2.putText(color, "MOCK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 200, 255), 2, cv2.LINE_AA)

        self.frame_no += 1
        return Frame(color, depth, depth_vis, t * 1000.0, self.frame_no,
                     time.time())

    def metadata(self):
        return {
            "device": self.device_name, "serial_number": self.serial,
            "firmware_version": self.firmware,
            "stream": {"width": WIDTH, "height": HEIGHT, "fps": FPS,
                       "depth_format": "Z16"},
            "depth_scale_m_per_unit": self.depth_scale,
            "depth_intrinsics": {"note": "mock camera, no real intrinsics"},
        }

    def stop(self):
        pass


# ---------------------------- recording ----------------------------
class RecordingSession:
    """One recording: a writer thread drains a frame queue into 16-bit PNGs
    plus a timestamp CSV, then finalizes metadata and renames the folder to
    include the duration."""

    def __init__(self, root_dir, camera_meta, labels=None):
        self.camera_meta = camera_meta
        self.labels = list(labels) if labels else ["unlabeled"]
        self.label_counts = {name: 0 for name in self.labels}
        self.events = []              # (elapsed_s, frame_index, kind, value)
        self._events_lock = threading.Lock()
        self.root_dir = root_dir
        self.start_wall = time.time()
        self.start_iso = now_iso()
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = root_dir / stamp
        n = 2
        while self.dir.exists():
            self.dir = root_dir / f"{stamp}_{n}"
            n += 1
        (self.dir / "depth").mkdir(parents=True)

        self.queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.saved = 0
        self.dropped = 0
        self.write_failures = 0
        self.writer_error = None
        self.active = True            # capture thread reads this
        self.state = "recording"      # recording -> saving -> done
        self.duration = 0.0
        self.result_msg = ""
        self._writer_thread = threading.Thread(target=self._writer, daemon=True)
        self._writer_thread.start()

    def add_frame(self, depth, ts_ms, host_time, label_idx=0, trial_id=0):
        try:
            self.queue.put_nowait((depth, ts_ms, host_time, label_idx,
                                   trial_id))
        except queue.Full:
            self.dropped += 1

    def log_event(self, kind, value=""):
        """Record a stimulus / label / trial marker with its elapsed time."""
        with self._events_lock:
            self.events.append((time.time() - self.start_wall, self.saved,
                                kind, str(value)))

    def stop(self):
        if not self.active:
            return
        self.active = False
        self.duration = time.time() - self.start_wall
        self.end_iso = now_iso()
        self.state = "saving"
        threading.Thread(target=self._finalize, daemon=True).start()

    # -- worker threads --
    def _writer(self):
        try:
            with open(self.dir / "frame_timestamps.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame_index", "filename", "device_timestamp_ms",
                            "host_time_unix", "host_time_iso", "label",
                            "trial_id"])
                while True:
                    item = self.queue.get()
                    if item is None:
                        return
                    depth, ts_ms, host_time, label_idx, trial_id = item
                    name = f"depth/depth_{self.saved:06d}.png"
                    # imencode + open('wb') instead of cv2.imwrite: reports
                    # failures and handles non-ASCII Windows paths
                    try:
                        ok, buf = cv2.imencode(
                            ".png", depth,
                            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
                        if ok:
                            with open(self.dir / name, "wb") as pf:
                                pf.write(buf.tobytes())
                    except (cv2.error, OSError):
                        ok = False
                    if not ok:
                        self.write_failures += 1
                        continue
                    label = self.labels[label_idx] \
                        if 0 <= label_idx < len(self.labels) else "unlabeled"
                    w.writerow([self.saved, name, f"{ts_ms:.3f}",
                                f"{host_time:.6f}",
                                dt.datetime.fromtimestamp(host_time)
                                .isoformat(timespec="milliseconds"),
                                label, trial_id])
                    self.label_counts[label] = \
                        self.label_counts.get(label, 0) + 1
                    self.saved += 1
        except Exception as e:
            self.writer_error = f"{type(e).__name__}: {e}"
            # Keep draining so add_frame and the stop sentinel never block
            # on a full queue with no consumer.
            while self.queue.get() is not None:
                self.write_failures += 1

    def _finalize(self):
        try:
            deadline = time.time() + 30
            while self._writer_thread.is_alive() and time.time() < deadline:
                try:
                    self.queue.put(None, timeout=0.5)  # flush-and-stop sentinel
                    break
                except queue.Full:
                    pass
            self._writer_thread.join(timeout=30)

            suffix = f"_dur{self.duration:.1f}s"
            final_dir = self.dir.with_name(self.dir.name + suffix)
            n = 2
            while final_dir.exists():
                final_dir = self.dir.with_name(f"{self.dir.name}{suffix}_{n}")
                n += 1
            rename_note = ""
            try:
                self.dir.rename(final_dir)
                self.dir = final_dir
            except OSError as e:
                rename_note = (f"  [folder kept as {self.dir.name}; "
                               f"rename failed: {e}]")

            with self._events_lock:
                events = list(self.events)
            with open(self.dir / "events.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["elapsed_s", "frame_index", "kind", "value"])
                for elapsed, frame_idx, kind, value in events:
                    w.writerow([f"{elapsed:.3f}", frame_idx, kind, value])

            meta = {
                "program": "D405 Face Depth Recorder",
                "format_version": FORMAT_VERSION,
                "labels": self.labels,
                "label_frame_counts": self.label_counts,
                "start_time": self.start_iso,
                "end_time": self.end_iso,
                "duration_seconds": round(self.duration, 3),
                "frames_saved": self.saved,
                "frames_dropped": self.dropped,
                "frames_failed_to_write": self.write_failures,
                "effective_fps": round(self.saved / self.duration, 2)
                                 if self.duration > 0 else 0,
                "depth_files": "depth/depth_NNNNNN.png, 16-bit; "
                               "meters = pixel_value * depth_scale_m_per_unit",
                "camera": self.camera_meta,
            }
            if self.writer_error:
                meta["writer_error"] = self.writer_error
            with open(self.dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            msg = (f"Saved {self.saved} frames ({self.duration:.1f} s) "
                   f"→ {self.dir.name}{rename_note}")
            if self.dropped:
                msg += f"  [WARNING: {self.dropped} frames dropped]"
            if self.write_failures:
                msg += (f"  [WARNING: {self.write_failures} frames "
                        f"failed to write]")
            if self.writer_error:
                msg = f"[WRITER ERROR: {self.writer_error}]  " + msg

            try:
                log = self.root_dir / "recordings_log.csv"
                new = not log.exists()
                with open(log, "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["start_time", "folder", "duration_s",
                                    "frames_saved", "frames_dropped"])
                    w.writerow([self.start_iso, self.dir.name,
                                f"{self.duration:.3f}", self.saved,
                                self.dropped])
            except OSError as e:
                msg += f"  [could not update recordings_log.csv: {e}]"

            self.result_msg = msg
        except Exception as e:
            self.result_msg = (f"ERROR while finalizing: {type(e).__name__}: "
                               f"{e} — depth frames are in {self.dir}")
        finally:
            self.state = "done"    # never leave the app stuck on "saving"


# ---------------------------- GUI ----------------------------
class App:
    def __init__(self, root, source):
        self.root = root
        self.source = source
        self.lock = threading.Lock()
        self.latest = None
        self.cam_error = None
        self.recording = None
        self.stop_event = threading.Event()
        self.closing = False
        self._fps_times = collections.deque(maxlen=60)
        self.labels = load_labels()
        self.label_idx = 0        # int reads/writes are atomic in CPython, so
        self.trial_id = 0         # the capture thread can read these directly
        self._close_deadline = 0.0

        RECORD_ROOT.mkdir(exist_ok=True)
        self._build_ui()

        self.capture_thread = threading.Thread(target=self._capture_loop,
                                               daemon=True)
        self.capture_thread.start()
        self.root.after(33, self._tick)

    # -- UI construction --
    def _build_ui(self):
        r = self.root
        r.title("D405 Face Depth Recorder")
        r.configure(bg=BG)
        r.resizable(False, False)

        blank = np.zeros((DISP_H, DISP_W, 3), np.uint8)
        self._photo_l = self._to_photo(blank)
        self._photo_r = self._to_photo(blank)

        left_name = {"color": "Color stream",
                     "infrared": "Infrared stream (no color mode)",
                     "none": "No image stream"}[self.source.left_kind]
        tk.Label(r, text=left_name, bg=BG, fg=DIM).grid(
            row=0, column=0, pady=(8, 2))
        tk.Label(r, text="Depth stream", bg=BG, fg=DIM).grid(
            row=0, column=1, pady=(8, 2))

        self.panel_l = tk.Label(r, image=self._photo_l, bg="black", bd=0)
        self.panel_l.grid(row=1, column=0, padx=(10, 5))
        self.panel_r = tk.Label(r, image=self._photo_r, bg="black", bd=0)
        self.panel_r.grid(row=1, column=1, padx=(5, 10))

        bar = tk.Frame(r, bg=BG)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=8)

        self.btn_start = tk.Button(
            bar, text="●  Start Recording", command=self.start_recording,
            bg=GREEN, fg="white", activebackground=GREEN, relief="flat",
            font=("Segoe UI", 11, "bold"), padx=16, pady=6, takefocus=0)
        self.btn_start.pack(side="left")
        self.btn_stop = tk.Button(
            bar, text="■  Stop", command=self.stop_recording,
            bg="#333844", fg="white", activebackground=RED, relief="flat",
            font=("Segoe UI", 11, "bold"), padx=16, pady=6, state="disabled",
            takefocus=0)
        self.btn_stop.pack(side="left", padx=8)

        self.var_rec = tk.StringVar(value="idle")
        tk.Label(bar, textvariable=self.var_rec, bg=BG, fg=RED,
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=16)

        # -- label bar: tags every frame as it is recorded (training labels) --
        lab = tk.Frame(r, bg=BG)
        lab.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10)
        tk.Label(lab, text="Label:", bg=BG, fg=DIM,
                 font=("Segoe UI", 9)).pack(side="left")
        self.label_buttons = []
        for i, name in enumerate(self.labels[:10]):
            key = "0" if i == 9 else str(i + 1)
            # takefocus=0 throughout: otherwise a click leaves focus on the
            # button and Space then both toggles recording AND re-fires it
            b = tk.Button(lab, text=f"{key} {name}", relief="flat",
                          font=("Segoe UI", 9), padx=6, pady=2, takefocus=0,
                          command=lambda idx=i: self.set_label(idx))
            b.pack(side="left", padx=2)
            self.label_buttons.append(b)
        if len(self.labels) > 10:
            tk.Label(lab, text=f"(+{len(self.labels) - 10} more, "
                               f"no hotkey)", bg=BG, fg=DIM,
                     font=("Segoe UI", 8)).pack(side="left", padx=4)
        self.btn_trial = tk.Button(
            lab, text="T  trial start", relief="flat", font=("Segoe UI", 9),
            padx=6, pady=2, bg="#333844", fg="white", takefocus=0,
            command=self.toggle_trial)
        self.btn_trial.pack(side="left", padx=(14, 2))
        self._refresh_label_buttons()

        self.var_msg = tk.StringVar(value="")
        tk.Label(r, textvariable=self.var_msg, bg=BG, fg=FG,
                 font=("Segoe UI", 9)).grid(row=4, column=0, columnspan=2,
                                            sticky="w", padx=12, pady=(6, 0))

        self.var_info = tk.StringVar(value="starting camera...")
        tk.Label(r, textvariable=self.var_info, bg=BG, fg=DIM,
                 font=("Segoe UI", 9)).grid(row=5, column=0, columnspan=2,
                                            sticky="w", padx=12)

        tk.Label(r, text="Space: start/stop    R: start    S: stop    "
                         "1-9/0: set label    T: trial marker    Q/Esc: quit",
                 bg=BG, fg=DIM,
                 font=("Segoe UI", 9)).grid(row=6, column=0, columnspan=2,
                                            sticky="w", padx=12, pady=(0, 8))

        for i in range(min(len(self.labels), 10)):
            key = "0" if i == 9 else str(i + 1)
            r.bind(key, lambda e, idx=i: self.set_label(idx))
        r.bind("t", lambda e: self.toggle_trial())
        r.bind("T", lambda e: self.toggle_trial())

        r.bind("<space>", lambda e: self.toggle_recording())
        r.bind("r", lambda e: self.start_recording())
        r.bind("R", lambda e: self.start_recording())
        r.bind("s", lambda e: self.stop_recording())
        r.bind("S", lambda e: self.stop_recording())
        r.bind("q", lambda e: self.on_close())
        r.bind("Q", lambda e: self.on_close())
        r.bind("<Escape>", lambda e: self.on_close())
        r.protocol("WM_DELETE_WINDOW", self.on_close)

    @staticmethod
    def _to_photo(img_bgr):
        ok, buf = cv2.imencode(".ppm", img_bgr)
        return tk.PhotoImage(data=buf.tobytes())

    # -- capture thread --
    def _capture_loop(self):
        while not self.stop_event.is_set():
            try:
                f = self.source.read()
            except Exception as e:
                with self.lock:
                    self.cam_error = str(e)
                time.sleep(0.5)
                continue
            if f is None:
                continue
            with self.lock:
                self.latest = f
                self.cam_error = None
                rec = self.recording
            self._fps_times.append(f.host_time)
            if rec is not None and rec.active:
                rec.add_frame(f.depth, f.ts_ms, f.host_time,
                              self.label_idx, self.trial_id)

    # -- labelling (main thread) --
    def set_label(self, idx):
        if not 0 <= idx < len(self.labels):
            return
        self.label_idx = idx
        self._refresh_label_buttons()
        rec = self.recording
        if rec is not None and rec.active:
            rec.log_event("label", self.labels[idx])

    def toggle_trial(self):
        rec = self.recording
        if self.trial_id == 0:
            self.trial_id = getattr(self, "_next_trial", 1)
            self._next_trial = self.trial_id + 1
            self.btn_trial.configure(text=f"T  trial {self.trial_id} ▪ end",
                                     bg=GREEN)
            if rec is not None and rec.active:
                rec.log_event("trial_start", self.trial_id)
        else:
            if rec is not None and rec.active:
                rec.log_event("trial_end", self.trial_id)
            self.trial_id = 0
            self.btn_trial.configure(text="T  trial start", bg="#333844")

    def _refresh_label_buttons(self):
        for i, b in enumerate(self.label_buttons):
            active = (i == self.label_idx)
            b.configure(bg="#3b6cd4" if active else "#333844",
                        fg="white",
                        font=("Segoe UI", 9, "bold" if active else "normal"))

    # -- recording control (main thread) --
    def start_recording(self):
        if self.recording is not None or self.closing:
            return
        try:
            session = RecordingSession(RECORD_ROOT, self.source.metadata(),
                                       labels=self.labels)
        except OSError as e:
            self.var_msg.set(f"Cannot create recording folder: {e}")
            return
        session.log_event("label", self.labels[self.label_idx])
        if self.trial_id:
            session.log_event("trial_start", self.trial_id)
        with self.lock:
            self.recording = session
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.var_msg.set(f"Recording to {session.dir.name} ...")

    def stop_recording(self):
        rec = self.recording
        if rec is None or not rec.active:
            return
        rec.stop()
        self.btn_stop.configure(state="disabled")

    def toggle_recording(self):
        if self.recording is None:
            self.start_recording()
        elif self.recording.active:
            self.stop_recording()

    # -- GUI update loop --
    def _tick(self):
        with self.lock:
            f = self.latest
            err = self.cam_error
            rec = self.recording

        if f is not None:
            if f.color is not None:
                left = cv2.resize(f.color, (DISP_W, DISP_H))
            else:
                left = np.zeros((DISP_H, DISP_W, 3), np.uint8)
                cv2.putText(left, "no image stream", (20, DISP_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128),
                            1, cv2.LINE_AA)
            right = cv2.resize(f.depth_vis, (DISP_W, DISP_H))
            self._draw_crosshair(right, f.depth)
            if rec is not None and rec.state == "recording" \
                    and int(time.time() * 2) % 2 == 0:
                for img in (left, right):
                    cv2.circle(img, (18, 18), 8, (60, 60, 229), -1)
            tag = self.labels[self.label_idx]
            if self.trial_id:
                tag += f"  |  trial {self.trial_id}"
            cv2.putText(left, tag, (34, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(left, tag, (34, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (120, 230, 120) if self.label_idx else (200, 200, 200),
                        1, cv2.LINE_AA)
            self._photo_l = self._to_photo(left)
            self._photo_r = self._to_photo(right)
            self.panel_l.configure(image=self._photo_l)
            self.panel_r.configure(image=self._photo_r)

        if err:
            self.var_info.set(f"CAMERA ERROR: {err}")
        elif f is not None:
            fps = 0.0
            if len(self._fps_times) >= 2:
                span = self._fps_times[-1] - self._fps_times[0]
                if span > 0:
                    fps = (len(self._fps_times) - 1) / span
            self.var_info.set(
                f"{self.source.device_name}  S/N {self.source.serial}   "
                f"{WIDTH}x{HEIGHT} @ {fps:.1f} fps   "
                f"depth unit {self.source.depth_scale * 1000:.3g} mm")

        if rec is not None:
            if rec.state == "recording":
                el = time.time() - rec.start_wall
                dot = "● " if int(time.time() * 2) % 2 == 0 else "   "
                self.var_rec.set(f"{dot}REC {int(el // 60):02d}:{el % 60:04.1f}"
                                 f"   {rec.saved} frames")
            elif rec.state == "saving":
                self.var_rec.set("saving...")
            elif rec.state == "done":
                self.var_rec.set("idle")
                self.var_msg.set(rec.result_msg)
                with self.lock:
                    self.recording = None
                self.btn_start.configure(state="normal")
                self.btn_stop.configure(state="disabled")
        else:
            self.var_rec.set("idle")

        if not self.closing:
            self.root.after(33, self._tick)

    def _draw_crosshair(self, img, depth_raw):
        cx, cy = DISP_W // 2, DISP_H // 2
        cv2.line(img, (cx - 12, cy), (cx + 12, cy), (255, 255, 255), 1)
        cv2.line(img, (cx, cy - 12), (cx, cy + 12), (255, 255, 255), 1)
        h, w = depth_raw.shape
        win = depth_raw[h // 2 - 4:h // 2 + 5, w // 2 - 4:w // 2 + 5]
        valid = win[win > 0]
        if valid.size:
            mm = float(np.median(valid)) * self.source.depth_scale * 1000.0
            text = f"center: {mm:.0f} mm"
        else:
            text = "center: --"
        cv2.putText(img, text, (cx + 18, cy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # -- shutdown --
    def on_close(self):
        if self.closing:
            return
        rec = self.recording
        if rec is not None and rec.active:
            rec.stop()
        self.closing = True
        self._close_deadline = time.time() + 90
        self._wait_close()

    def _wait_close(self):
        rec = self.recording
        if (rec is not None and rec.state == "saving"
                and time.time() < self._close_deadline):
            self.var_rec.set("saving... (closing when done)")
            self.root.after(100, self._wait_close)
            return
        self.stop_event.set()
        self.capture_thread.join(timeout=3)
        self.source.stop()
        self.root.destroy()


# ---------------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser(description="D405 face depth recorder")
    ap.add_argument("--mock", action="store_true",
                    help="run with a synthetic camera (no hardware needed)")
    args = ap.parse_args()

    if args.mock:
        source = MockSource()
    else:
        try:
            source = RealSenseSource()
        except RuntimeError as e:
            root = tk.Tk()
            root.withdraw()
            from tkinter import messagebox
            messagebox.showerror(
                "D405 Face Depth Recorder",
                f"Could not open the RealSense D405:\n\n{e}\n\n"
                "Check the USB 3 connection, close other RealSense apps "
                "(e.g. RealSense Viewer), or run with --mock to test the GUI.")
            sys.exit(1)

    root = tk.Tk()
    App(root, source)
    root.mainloop()


if __name__ == "__main__":
    main()
