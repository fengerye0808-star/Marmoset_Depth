"""
Canonical depth preprocessing for the D405 face recordings.

This module is the SINGLE definition of how a raw 16-bit depth frame becomes
a model input. It is imported by

    build_dataset.py   (offline: recordings -> training tensors)
    depth_dataset.py   (training-time batching)
    live_infer.py      (real-time inference)

so training and deployment can never drift apart. It is pure computation --
no file I/O, no camera, no torch -- which also makes it easy to unit test.

Pipeline (PREPROC_VERSION v2)
-----------------------------
Fitted once per session (or per live warm-up), from the neutral period:

1. Keep only pixels within `search_band_m` of `expected_distance_m` -- i.e.
   a thin shell around where you positioned the face. This is what keeps the
   torso, the headbar and the lickspout out of the estimate; a plain 6-40 cm
   range mask does NOT, and in a real head-fixed rig it silently lands the
   reference on the animal's body.
2. Require a pixel to be in that band in most reference frames (temporally
   stable), then take the LARGEST CONNECTED COMPONENT. The face is one
   coherent surface; a spout or a headbar edge is not connected to it.
3. ROI = that component's box, squared, padded, and capped to a plausible
   size. Held fixed for the session (valid because the rig is head-fixed).
4. reference_mm = median depth of that component only.
5. Refuse the fit if the reference is not within `ref_tolerance_m` of
   `expected_distance_m`. A reference on the torso is now an error, not
   silent data corruption.

Per frame:

6. Crop the ROI, convert to depth relative to reference_mm, in millimetres.
7. Pixels outside +/- clip_mm are marked INVALID (NaN), not clamped to the
   rail. Clamping invents a flat surface where there is no usable measurement;
   NaN says "no data" and shows up honestly in the valid-fraction QC.
8. Masked area-resize to out_size x out_size (invalid pixels never bleed into
   valid ones).

Storage stays in unclipped millimetres (physically meaningful, and clipping
before a difference would destroy deformation -- see `to_model_input`).

`to_model_input` does the final scaling and builds the validity mask channel,
and is likewise shared between training and inference.

Two training views come free from the same stored array:
    mode="absolute" -- relative depth, keeps the subject's static face shape
    mode="deform"   -- minus the session's neutral reference image, leaving
                       pure facial deformation (usually the stronger signal
                       for expression, and subject-invariant)
"""

import warnings

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("preprocessing.py needs opencv-python") from e

PREPROC_VERSION = "v2"

DEFAULT_PARAMS = {
    "out_size": 128,             # network input is out_size x out_size
    "expected_distance_m": 0.15,  # where you positioned the face
    "search_band_m": 0.06,       # ROI/reference search: expected +/- this
    "ref_tolerance_m": 0.05,     # fit fails if the reference lands outside this
    "clip_mm": 40.0,             # representable relative-depth range, +/- mm
    "deform_clip_mm": 20.0,      # scaling range for deform mode
    "roi_margin": 1.25,          # pad the face component's box by this factor
    "max_roi_frac": 1.0,         # ROI side <= this * min(frame h, w)
    "min_relief_mm": 3.0,        # a face has relief; a headbar/wall does not
    "min_component_frac": 0.001,  # smallest acceptable face component
    "roi_override": None,        # [x, y, side]: skip ROI estimation entirely
    "stable_frac": 0.5,          # pixel must be in band in this frac of frames
    "coverage_thresh": 0.25,     # output pixel needs this much valid support
    "per_frame_dc": "none",      # "none" | "roi_median" (see fit_notes below)
}

PER_FRAME_DC_CHOICES = ("none", "roi_median")


# ---------------------------------------------------------------- helpers
def square_roi(bbox, shape, margin, max_frac=1.0):
    """Grow bbox to a margin-padded square that fits inside `shape` and is no
    larger than max_frac of the short side. Returns (x, y, side)."""
    h, w = shape
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = int(round(max(x1 - x0, y1 - y0) * margin))
    side = min(side, int(round(min(h, w) * max_frac)), h, w)
    side = max(8, side)
    x = max(0, min(int(round(cx - side / 2.0)), w - side))
    y = max(0, min(int(round(cy - side / 2.0)), h - side))
    return x, y, side


def largest_component(mask, min_pixels):
    """(bbox, component_mask, n_pixels) of the largest connected blob, or
    (None, None, 0). Connectivity is what separates the face from a spout or
    headbar that merely happens to sit at a similar depth."""
    m = mask.astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    best, best_area = 0, 0
    for i in range(1, n):                       # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best, best_area = i, area
    if best == 0 or best_area < min_pixels:
        return None, None, best_area
    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    w = int(stats[best, cv2.CC_STAT_WIDTH])
    h = int(stats[best, cv2.CC_STAT_HEIGHT])
    return (x, y, x + w, y + h), (lbl == best), best_area


def masked_resize(data, valid, out_size, coverage_thresh):
    """Area-resize `data` using only valid pixels. Output pixels whose valid
    support is below coverage_thresh become NaN. Prevents holes and the
    background from bleeding into the face during downsampling."""
    d = np.where(valid, data, 0.0).astype(np.float32)
    m = valid.astype(np.float32)
    size = (out_size, out_size)
    num = cv2.resize(d, size, interpolation=cv2.INTER_AREA)
    den = cv2.resize(m, size, interpolation=cv2.INTER_AREA)
    out = np.full(size, np.nan, np.float32)
    good = den > coverage_thresh
    out[good] = num[good] / den[good]
    return out


def frame_stats(rel_mm, clip_mm):
    """QC numbers for one preprocessed frame. `rail_frac` catches the failure
    where everything saturates at the clip limit and looks like a valid flat
    surface; `std_mm` catches an information-free (constant) frame."""
    fin = np.isfinite(rel_mm)
    n_fin = int(fin.sum())
    if not n_fin:
        return {"valid_frac": 0.0, "rail_frac": 1.0, "std_mm": 0.0,
                "median_mm": float("nan")}
    v = rel_mm[fin]
    return {
        "valid_frac": n_fin / rel_mm.size,
        "rail_frac": float(np.mean(np.abs(v) >= clip_mm - 1e-3)),
        "std_mm": float(np.std(v)),
        "median_mm": float(np.median(v)),
    }


# ---------------------------------------------------------------- main class
class DepthPreprocessor:
    """Holds the per-session state (ROI + reference depth) and applies the
    canonical transform. Fit once per recording / per live warm-up, then call
    `transform` on every frame."""

    def __init__(self, depth_scale, **params):
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f"unknown preprocessing params: {sorted(unknown)}")
        self.depth_scale = float(depth_scale)
        self.params = {**DEFAULT_PARAMS, **params}
        if self.params["per_frame_dc"] not in PER_FRAME_DC_CHOICES:
            raise ValueError(f"per_frame_dc must be one of "
                             f"{PER_FRAME_DC_CHOICES}")
        ro = self.params["roi_override"]
        if ro is not None:
            ro = [int(v) for v in ro]
            if len(ro) != 3:
                raise ValueError("roi_override must be [x, y, side]")
            self.params["roi_override"] = ro
        self.roi = None               # (x, y, side) in raw-frame pixels
        self.reference_mm = None      # the session's neutral face distance
        self.reference_image = None   # (out, out) neutral surface, relative mm
        self.n_reference_frames = 0
        self.fit_notes = {}           # diagnostics for QC / dataset spec

    def __getattr__(self, name):
        # expose params (out_size, clip_mm, ...) as attributes
        params = self.__dict__.get("params")
        if params and name in params:
            return params[name]
        raise AttributeError(name)

    @property
    def is_fitted(self):
        return self.roi is not None and self.reference_mm is not None

    # -- fitting --
    def fit(self, frames):
        """Estimate the session ROI and reference depth from warm-up frames.

        frames: iterable of raw uint16 depth arrays (same shape), from the
        NEUTRAL period of the recording.

        Raises ValueError with an actionable message if no plausible face
        surface is found at the expected distance -- far better than fitting
        to the animal's torso and silently flattening every expression.
        """
        stack = np.asarray([np.asarray(f) for f in frames])
        if stack.ndim != 3 or stack.shape[0] == 0:
            raise ValueError("fit() needs a non-empty sequence of 2-D frames")
        self.n_reference_frames = int(stack.shape[0])
        h, w = stack.shape[1:]

        depth_m = stack.astype(np.float32) * self.depth_scale
        exp, band = self.expected_distance_m, self.search_band_m
        near, far = exp - band, exp + band
        in_band = (depth_m > near) & (depth_m < far)

        # Temporally stable band membership: rejects transient speckle without
        # rejecting a real surface that has occasional stereo dropouts.
        stable = in_band.mean(axis=0) >= self.stable_frac
        min_px = max(32, int(self.min_component_frac * h * w))

        if self.roi_override:
            # Explicit ROI: the right choice when the face cannot be separated
            # automatically -- e.g. a mouse, whose head and body are contiguous
            # in depth so no connectivity rule can split them. The rig is fixed,
            # so a hand-measured box is both correct and perfectly reproducible.
            ox, oy, os_ = (int(v) for v in self.roi_override)
            if not (0 <= ox and 0 <= oy and ox + os_ <= w and oy + os_ <= h
                    and os_ >= 8):
                raise ValueError(
                    f"roi_override {list(self.roi_override)} does not fit "
                    f"inside a {w}x{h} frame")
            self.roi = (ox, oy, os_)
            comp = np.zeros((h, w), bool)
            comp[oy:oy + os_, ox:ox + os_] = stable[oy:oy + os_, ox:ox + os_]
            area = int(comp.sum())
            if area < min_px:
                raise ValueError(
                    f"only {area} pixels inside roi_override are at "
                    f"{exp*100:.0f} +/- {band*100:.0f} cm (need {min_px}). "
                    f"Check the ROI box and the subject distance.")
            bbox = (ox, oy, ox + os_, oy + os_)
        else:
            bbox, comp, area = largest_component(stable, min_px)
        if bbox is None:
            valid_any = depth_m[(depth_m > 0.02) & (depth_m < 1.0)]
            observed = (f"{np.percentile(valid_any, 5)*100:.1f}-"
                        f"{np.percentile(valid_any, 95)*100:.1f} cm"
                        if valid_any.size else "no depth at all")
            raise ValueError(
                f"no face-sized surface found at "
                f"{exp*100:.0f} +/- {band*100:.0f} cm "
                f"(largest candidate was {area} px, need {min_px}). "
                f"Depth in view spans {observed}. Reposition the subject to "
                f"~{exp*100:.0f} cm, or set expected_distance_m / "
                f"search_band_m to match your rig.")

        if not self.roi_override:
            self.roi = square_roi(bbox, (h, w), self.roi_margin,
                                  self.max_roi_frac)
        x, y, s = self.roi

        # Reference from the face component ONLY -- never the whole ROI, whose
        # median can sit on the torso when the ROI is large.
        comp_depths = depth_m[:, comp]
        comp_depths = comp_depths[(comp_depths > near) & (comp_depths < far)]
        self.reference_mm = float(np.median(comp_depths) * 1000.0)

        if abs(self.reference_mm / 1000.0 - exp) > self.ref_tolerance_m:
            raise ValueError(
                f"fitted reference depth {self.reference_mm:.0f} mm is more "
                f"than {self.ref_tolerance_m*1000:.0f} mm from the expected "
                f"{exp*1000:.0f} mm. The estimate probably locked onto the "
                f"wrong surface (body / headbar / spout). Check the subject "
                f"distance, or pass expected_distance_m for your rig.")

        # Per-pixel median over the warm-up -> the neutral surface. Pixels with
        # no valid sample are NaN by design, so silence the all-NaN warning.
        rep = np.where((depth_m > 0) & (np.abs(depth_m * 1000.0
                                               - self.reference_mm)
                                        <= self.clip_mm), depth_m, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med_frame = np.nanmedian(rep, axis=0)
        roi_med = med_frame[y:y + s, x:x + s]
        roi_valid = np.isfinite(roi_med)
        # NOT clipped: clipping before the deform difference would annihilate
        # deformation wherever the static surface is near the limit.
        self.reference_image = masked_resize(
            roi_med * 1000.0 - self.reference_mm, roi_valid,
            self.out_size, self.coverage_thresh)

        ref_valid = float(np.isfinite(self.reference_image).mean())
        comp_med = med_frame[comp]
        comp_med = comp_med[np.isfinite(comp_med)] * 1000.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            temporal_std = float(np.nanmedian(
                np.nanstd(rep[:, comp] * 1000.0, axis=0)))
        # Robust relief: a face is a curved surface, a headbar or a wall is
        # flat. Percentiles rather than min/max so speckle can't fake relief.
        relief = (float(np.percentile(comp_med, 95)
                        - np.percentile(comp_med, 5))
                  if comp_med.size else 0.0)
        if relief < self.min_relief_mm:
            raise ValueError(
                f"the surface found at {self.reference_mm:.0f} mm is flat "
                f"({relief:.1f} mm of relief, need {self.min_relief_mm:.0f}). "
                f"That looks like a headbar, wall or holder rather than a "
                f"face. Check what is at ~{exp*100:.0f} cm from the camera.")

        self.fit_notes = {
            "component_pixels": int(area),
            "component_frac_of_frame": round(area / (h * w), 4),
            "roi_side_px": int(s),
            "reference_valid_frac": round(ref_valid, 4),
            "reference_temporal_std_mm": round(temporal_std, 3),
            "surface_relief_mm": round(relief, 2),
            "frame_shape": [int(h), int(w)],
        }
        return self

    # -- per-frame transform --
    def transform(self, raw):
        """raw uint16 depth frame -> (out_size, out_size) float32 relative
        depth in mm, NaN where there is no usable depth. Values outside
        +/- clip_mm are NaN, not clamped."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before transform()")
        raw = np.asarray(raw)
        if raw.ndim != 2:
            raise ValueError(f"expected a 2-D depth frame, got {raw.shape}")
        exp_shape = self.fit_notes.get("frame_shape")
        if exp_shape and list(raw.shape) != exp_shape:
            raise ValueError(
                f"frame shape {list(raw.shape)} does not match the fitted "
                f"{exp_shape}: the ROI would point at the wrong pixels. Refit "
                f"on this stream's resolution.")
        x, y, s = self.roi
        roi = raw[y:y + s, x:x + s].astype(np.float32) * self.depth_scale
        rel_mm = roi * 1000.0 - self.reference_mm
        valid = (roi > 0) & (np.abs(rel_mm) <= self.clip_mm)

        if self.per_frame_dc == "roi_median" and valid.any():
            # Removes slow whole-face distance drift, which otherwise becomes a
            # per-label DC offset (labels are contiguous time blocks).
            rel_mm = rel_mm - float(np.median(rel_mm[valid]))
            valid = (roi > 0) & (np.abs(rel_mm) <= self.clip_mm)

        return masked_resize(rel_mm, valid, self.out_size,
                             self.coverage_thresh)

    # -- serialization: travels inside the dataset spec and the checkpoint --
    def state_dict(self):
        return {
            "preproc_version": PREPROC_VERSION,
            "depth_scale": self.depth_scale,
            "params": dict(self.params),
            "roi": list(self.roi) if self.roi else None,
            "reference_mm": self.reference_mm,
            "n_reference_frames": self.n_reference_frames,
            "fit_notes": dict(self.fit_notes),
            "reference_image": (None if self.reference_image is None
                                else self.reference_image.tolist()),
        }

    @classmethod
    def from_state_dict(cls, st, strict=True):
        if strict and st.get("preproc_version") != PREPROC_VERSION:
            raise ValueError(
                f"preprocessing version mismatch: data/checkpoint was built "
                f"with {st.get('preproc_version')}, this code is "
                f"{PREPROC_VERSION}. Rebuild the dataset or check out the "
                f"matching code.")
        obj = cls(st["depth_scale"], **st.get("params", {}))
        obj.roi = tuple(st["roi"]) if st.get("roi") else None
        obj.reference_mm = st.get("reference_mm")
        obj.n_reference_frames = st.get("n_reference_frames", 0)
        obj.fit_notes = dict(st.get("fit_notes", {}))
        ref = st.get("reference_image")
        obj.reference_image = (None if ref is None
                               else np.asarray(ref, np.float32))
        return obj


# ------------------------------------------------- final model-input step
def input_scale(params, mode):
    """The millimetre range that maps to [-1, 1] for a given mode. Every
    caller must use this so training and inference scale identically."""
    return float(params["deform_clip_mm"] if mode == "deform"
                 else params["clip_mm"])


def to_model_input(clip_mm, reference_image=None, clip_mm_max=None,
                   mode="absolute"):
    """Stored millimetre clip -> network input.

    clip_mm: (T, H, W) float array in relative mm, NaN = invalid.
    mode "absolute": relative depth as stored.
    mode "deform":   subtract the session neutral surface -> deformation only.
                     Pixels where EITHER the frame or the reference is missing
                     are invalid; reusing a hole as if it were zero would
                     silently mix absolute depth into a deformation map.

    Returns (2, T, H, W) float32: channel 0 depth scaled to [-1, 1] (0 where
    invalid), channel 1 the validity mask. The mask channel matters -- it lets
    the network distinguish "flat surface" from "no measurement".
    """
    clip = np.asarray(clip_mm, np.float32)
    if clip.ndim == 2:
        clip = clip[None]
    if clip_mm_max is None:
        clip_mm_max = DEFAULT_PARAMS["clip_mm"]

    mask = np.isfinite(clip)
    x = np.nan_to_num(clip, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "deform":
        if reference_image is None:
            raise ValueError("mode='deform' needs the session reference_image")
        ref = np.asarray(reference_image, np.float32)
        mask = mask & np.isfinite(ref)[None]
        x = x - np.nan_to_num(ref, nan=0.0)[None]
    elif mode != "absolute":
        raise ValueError(f"unknown mode {mode!r}")

    # Clip LAST, after any difference, so a difference is never taken between
    # two already-saturated values.
    x = np.clip(x / float(clip_mm_max), -1.0, 1.0) * mask
    return np.stack([x, mask.astype(np.float32)]).astype(np.float32)
