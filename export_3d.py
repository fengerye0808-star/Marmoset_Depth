#!/usr/bin/env python3
"""
Export recorded depth frames as metric 3D geometry (point clouds / meshes).

    python export_3d.py                              newest recording -> averaged face mesh
    python export_3d.py --average --mesh --out face.ply
    python export_3d.py --frame 120 --points --out frame120.ply
    python export_3d.py --sequence --every 3 --out-dir face_4d/
    python export_3d.py --label condition_1 --average --mesh --out cond1.ply

Why this works from "just PNGs"
-------------------------------
A 16-bit depth PNG plus the camera intrinsics IS a 3D surface -- it is an
*organised* point cloud, which is the compact lossless way to store one. Each
pixel (u, v) back-projects to a metric point:

    Z = raw * depth_scale                (metres)
    X = (u - ppx) * Z / fx
    Y = (v - ppy) * Z / fy

fx, fy, ppx, ppy come from `metadata.json -> camera.depth_intrinsics`, which the
recorder saves for every session. So nothing has to be re-recorded: this script
just makes the geometry explicit in a format 3D tools read.

Head-fixed means averaging is legitimate
----------------------------------------
Because the skull does not move, per-pixel temporal averaging over N frames of
one expression cuts depth noise by ~sqrt(N) without blurring the shape. That is
what `--average` does, and it is the difference between a grainy single-frame
scan and a clean face model. Use it with `--label neutral` for a canonical
neutral face, or with any other label for that expression's shape.

Output
------
PLY (binary by default, ASCII with --ascii) or OBJ with --format obj. Both open
in MeshLab, CloudCompare, Blender, Open3D and most other 3D software.
Coordinates are in MILLIMETRES by default (a face is ~110 mm across, so mm keeps
the numbers readable); use --units m for metres. Axes are the camera's own:
+X right, +Y down, +Z away from the lens.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

from build_dataset import imread_u16, read_session
from preprocessing import DEFAULT_PARAMS, DepthPreprocessor

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------ intrinsics
def get_intrinsics(meta, override=None):
    """(fx, fy, ppx, ppy) for the depth stream, or a clear error."""
    if override:
        vals = [float(v) for v in override.split(",")]
        if len(vals) != 4:
            sys.exit("--intrinsics needs fx,fy,ppx,ppy")
        return tuple(vals)
    intr = (meta.get("camera") or {}).get("depth_intrinsics") or {}
    needed = ("fx", "fy", "ppx", "ppy")
    if not all(k in intr for k in needed):
        sys.exit(
            "this recording has no camera intrinsics in metadata.json "
            f"(found keys: {sorted(intr)}).\n"
            "Real D405 recordings always store them. Synthetic/mock recordings "
            "do not, so pass them explicitly, e.g.\n"
            "  --intrinsics 423.5,423.5,424.0,240.0")
    coeffs = intr.get("coeffs") or []
    if any(abs(float(c)) > 1e-6 for c in coeffs):
        print(f"note: intrinsics carry non-zero distortion coeffs {coeffs}; "
              f"this script uses the pinhole model, which is exact for the "
              f"D405's rectified depth stream but approximate otherwise.")
    return (float(intr["fx"]), float(intr["fy"]),
            float(intr["ppx"]), float(intr["ppy"]))


# ------------------------------------------------------------ geometry
def deproject(depth_m, intr, scale_out=1000.0, step=1):
    """Organised depth (H, W) in metres -> (H, W, 3) coordinates.

    `step` is the pixel stride if the map was decimated, so the true sensor
    pixel coordinates are used and the geometry stays metrically correct.
    Invalid pixels (NaN or <= 0) come back as NaN.
    """
    fx, fy, ppx, ppy = intr
    h, w = depth_m.shape
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    uu, vv = uu * step, vv * step
    z = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, np.nan)
    x = (uu - ppx) * z / fx
    y = (vv - ppy) * z / fy
    return np.stack([x, y, z], axis=-1) * scale_out


def spread_tolerance_mm(ref_mm, explicit=None, frac=0.05, floor=5.0):
    """How much per-pixel depth variation to tolerate before calling it motion.

    MUST scale with distance: stereo depth noise grows roughly with Z^2, and the
    D405 is specced at about 2% of range, so a fixed few-millimetre tolerance
    rejects ordinary sensor noise at 30+ cm -- discarding exactly the pixels
    averaging is meant to clean up. 5% of the working distance sits comfortably
    above the noise while still catching gross subject movement.
    """
    if explicit:
        return float(explicit)
    if not ref_mm:
        return max(floor, 6.0)
    return max(floor, frac * float(ref_mm))


def average_depth(frames_raw, depth_scale, min_frames=3, max_spread_mm=6.0):
    """Per-pixel temporal median of several frames -> a low-noise depth map.

    Returns (depth_m, n_frames, dropped_fraction).

    Pixels whose depth varies more than max_spread_mm across the window are
    dropped: that variation means the subject moved (or the stereo match is
    unstable), and averaging across it would invent a surface that was never
    there. This is why the averaging window must be SHORT unless the subject is
    genuinely rigid -- a freely moving subject over many seconds fails this test
    at almost every pixel, and the caller must be told rather than handed an
    empty map. See spread_tolerance_mm for why the threshold is distance-scaled.
    """
    stack = np.asarray(frames_raw, np.float32) * depth_scale
    stack[stack <= 0] = np.nan
    finite = np.isfinite(stack).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns
        med = np.nanmedian(stack, axis=0)
        lo = np.nanpercentile(stack, 10, axis=0)
        hi = np.nanpercentile(stack, 90, axis=0)
    had = np.isfinite(med)
    bad = (finite < min_frames) | ((hi - lo) * 1000.0 > max_spread_mm)
    med[bad] = np.nan
    n_had = int(had.sum())
    dropped = float((bad & had).sum() / n_had) if n_had else 1.0
    return med, int(stack.shape[0]), dropped


def suggest_distance(depth_m, lo_mm=50.0, hi_mm=900.0, bins=43):
    """Where is the subject actually? Returns the densest near-surface depth in
    mm, or None. Used to turn 'no valid points' into an actionable message."""
    d = np.asarray(depth_m, np.float32) * 1000.0
    v = d[np.isfinite(d) & (d > lo_mm) & (d < hi_mm)]
    if v.size < 500:
        return None, 0.0
    hist, edges = np.histogram(v, bins=bins, range=(lo_mm, hi_mm))
    k = int(np.argmax(hist))
    centre = float((edges[k] + edges[k + 1]) / 2)
    covered = float(np.mean(np.abs(v - centre) <= 80.0))
    return centre, covered


def keep_largest_surface(depth_m, min_pixels=200):
    """Keep only the biggest connected patch of valid depth.

    A crop box plus a depth band still admits anything that happens to sit near
    the face -- a lickspout, a headbar edge, the operator's hand. Those are
    separate surfaces, not part of the face, and connectivity removes them the
    same way the training pipeline's ROI fitter does.
    """
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not valid.any():
        return depth_m, 0
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(
        valid.astype(np.uint8), connectivity=8)
    best, best_area = 0, 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best, best_area = i, area
    if best == 0 or best_area < min_pixels:
        return depth_m, 0
    return np.where(lbl == best, depth_m, np.nan), n - 1


def build_mesh(points, max_edge_mm=4.0):
    """Triangulate an organised (H, W, 3) grid.

    Adjacent valid pixels are connected only when the triangle's edges are
    shorter than max_edge_mm, so the mesh does not stitch a sheet across the
    gap between the face and whatever is behind it.
    """
    h, w = points.shape[:2]
    valid = np.isfinite(points).all(axis=-1)
    idx = np.full((h, w), -1, np.int64)
    idx[valid] = np.arange(int(valid.sum()))
    verts = points[valid].astype(np.float32)

    tl, tr = idx[:-1, :-1], idx[:-1, 1:]
    bl, br = idx[1:, :-1], idx[1:, 1:]
    p_tl, p_tr = points[:-1, :-1], points[:-1, 1:]
    p_bl, p_br = points[1:, :-1], points[1:, 1:]

    def edge(a, b):
        return np.linalg.norm(a - b, axis=-1)

    tris = []
    for i0, i1, i2, q0, q1, q2 in (
            (tl, tr, bl, p_tl, p_tr, p_bl),
            (tr, br, bl, p_tr, p_br, p_bl)):
        ok = ((i0 >= 0) & (i1 >= 0) & (i2 >= 0)
              & (edge(q0, q1) < max_edge_mm)
              & (edge(q1, q2) < max_edge_mm)
              & (edge(q0, q2) < max_edge_mm))
        if ok.any():
            tris.append(np.stack([i0[ok], i1[ok], i2[ok]], axis=1))
    faces = (np.concatenate(tris, axis=0).astype(np.int32) if tris
             else np.zeros((0, 3), np.int32))
    return verts, faces


def vertex_normals(verts, faces):
    """Area-weighted vertex normals, so viewers shade the surface correctly."""
    normals = np.zeros_like(verts)
    if len(faces):
        a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        fn = np.cross(b - a, c - a)
        for k in range(3):
            np.add.at(normals, faces[:, k], fn)
    n = np.linalg.norm(normals, axis=1, keepdims=True)
    n[n == 0] = 1.0
    out = normals / n
    # camera looks down +Z, so face the normals back toward the lens
    flip = out[:, 2] > 0
    out[flip] *= -1
    return out.astype(np.float32)


def depth_colors(verts):
    """Pseudo-colour by depth, purely so the export is readable in a viewer."""
    z = verts[:, 2]
    lo, hi = np.percentile(z, [2, 98])
    t = np.clip((z - lo) / max(1e-6, hi - lo), 0, 1)
    bgr = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return bgr.reshape(-1, 3)[:, ::-1].copy()      # BGR -> RGB


# ------------------------------------------------------------ writers
def write_ply(path, verts, faces=None, normals=None, colors=None,
              binary=True, comments=()):
    n_v = len(verts)
    n_f = 0 if faces is None else len(faces)
    props = ["property float x", "property float y", "property float z"]
    if normals is not None:
        props += ["property float nx", "property float ny", "property float nz"]
    if colors is not None:
        props += ["property uchar red", "property uchar green",
                  "property uchar blue"]
    header = ["ply",
              f"format {'binary_little_endian' if binary else 'ascii'} 1.0"]
    header += [f"comment {c}" for c in comments]
    header += [f"element vertex {n_v}"] + props
    if n_f:
        header += [f"element face {n_f}",
                   "property list uchar int vertex_indices"]
    header += ["end_header"]
    head = ("\n".join(header) + "\n").encode("ascii")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        if normals is not None:
            fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
        if colors is not None:
            fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        arr = np.empty(n_v, dtype=fields)
        arr["x"], arr["y"], arr["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
        if normals is not None:
            arr["nx"], arr["ny"], arr["nz"] = (normals[:, 0], normals[:, 1],
                                               normals[:, 2])
        if colors is not None:
            arr["red"], arr["green"], arr["blue"] = (colors[:, 0], colors[:, 1],
                                                     colors[:, 2])
        with open(path, "wb") as f:
            f.write(head)
            f.write(arr.tobytes())
            if n_f:
                rec = np.empty(n_f, dtype=[("n", "u1"), ("v", "<i4", 3)])
                rec["n"] = 3
                rec["v"] = faces
                f.write(rec.tobytes())
    else:
        with open(path, "wb") as f:
            f.write(head)
            cols = [verts]
            if normals is not None:
                cols.append(normals)
            if colors is not None:
                cols.append(colors.astype(np.float32))
            data = np.hstack(cols)
            fmt = " ".join(["%.4f"] * (3 + (3 if normals is not None else 0))
                           + ["%d"] * (3 if colors is not None else 0))
            np.savetxt(f, data, fmt=fmt)
            if n_f:
                np.savetxt(f, np.hstack(
                    [np.full((n_f, 1), 3, np.int32), faces]), fmt="%d")
    return path


def read_ply(path):
    """Minimal reader for the PLY files this script writes -> (verts, faces).
    Handy for checking an export, or loading it into numpy without extra deps."""
    raw = Path(path).read_bytes()
    end = raw.find(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii").splitlines()
    binary = any("binary_little_endian" in h for h in header)
    n_v = n_f = 0
    props, element = [], None
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            element = parts[1]
            if element == "vertex":
                n_v = int(parts[2])
            elif element == "face":
                n_f = int(parts[2])
        elif parts[0] == "property" and element == "vertex":
            props.append((parts[2], parts[1]))
    np_type = {"float": "<f4", "uchar": "u1", "int": "<i4", "double": "<f8"}
    if binary:
        dt = np.dtype([(n, np_type[t]) for n, t in props])
        varr = np.frombuffer(raw, dtype=dt, count=n_v, offset=end)
        verts = np.stack([varr["x"], varr["y"], varr["z"]], axis=1)
        faces = None
        if n_f:
            off = end + varr.nbytes
            fdt = np.dtype([("n", "u1"), ("v", "<i4", 3)])
            farr = np.frombuffer(raw, dtype=fdt, count=n_f, offset=off)
            faces = np.asarray(farr["v"])
    else:
        rows = raw[end:].decode("ascii").split("\n")
        vals = np.array([[float(x) for x in r.split()]
                         for r in rows[:n_v] if r.strip()])
        verts = vals[:, :3]
        faces = (np.array([[int(x) for x in r.split()[1:4]]
                           for r in rows[n_v:n_v + n_f] if r.strip()])
                 if n_f else None)
    return verts.astype(np.float32), faces


def shade_preview(points, path, light=(-0.35, -0.55, -1.0)):
    """Render the surface as a shaded relief PNG.

    No 3D rasteriser needed: the depth map already IS an image-space surface, so
    per-pixel normals plus a diffuse term give an immediately readable picture of
    the shape. Useful for confirming at a glance that the geometry is a face.
    """
    z = points[..., 2].astype(np.float32)
    gy, gz = np.gradient(np.nan_to_num(z, nan=np.nanmedian(z)))
    # millimetres per pixel, so the normals have the right aspect
    with np.errstate(all="ignore"):
        px = float(np.nanmedian(np.abs(np.diff(points[..., 0], axis=1)))) or 1.0
        py = float(np.nanmedian(np.abs(np.diff(points[..., 1], axis=0)))) or 1.0
    nx, ny, nz = -gz / px, -gy / py, np.ones_like(z)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    lx, ly, lz = np.asarray(light, np.float32) / np.linalg.norm(light)
    diffuse = np.clip(-(nx * lx + ny * ly + nz * lz), 0, 1)
    shade = np.clip(0.18 + 0.85 * diffuse, 0, 1)
    img = (shade * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img[~np.isfinite(z)] = (24, 20, 18)
    h, w = img.shape[:2]
    if max(h, w) < 400:
        f = 400 // max(h, w) + 1
        img = cv2.resize(img, (w * f, h * f), interpolation=cv2.INTER_NEAREST)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if ok:
        path.write_bytes(buf.tobytes())
    return path


def write_obj(path, verts, faces=None, normals=None, comments=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        for c in comments:
            f.write(f"# {c}\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        if faces is not None:
            for t in faces + 1:                      # OBJ is 1-indexed
                if normals is not None:
                    f.write(f"f {t[0]}//{t[0]} {t[1]}//{t[1]} "
                            f"{t[2]}//{t[2]}\n")
                else:
                    f.write(f"f {t[0]} {t[1]} {t[2]}\n")
    return path


# ------------------------------------------------------------ frame selection
def iter_points(rec_dir, expected_distance_m=None, face_depth_mm=80.0,
                roi=None, every=1, units="mm", largest_only=True,
                label=None):
    """Yield (frame_index, label, points) for every frame of a recording.

        for i, lab, pts in iter_points("recordings/2026-...", expected_distance_m=0.32):
            xyz = pts[np.isfinite(pts).all(-1)]      # (N, 3) in mm

    `points` is the organised (H, W, 3) grid, NaN where there is no depth, so
    pixel neighbourhoods stay intact. This is the cheap way to work with the
    whole recording in 3D: exporting every frame to PLY re-encodes the same
    information at roughly 30x the size, which is only worth doing when another
    program has to read it.
    """
    rec_dir = Path(rec_dir)
    meta, rows = read_session(rec_dir)
    if meta is None:
        raise FileNotFoundError(f"{rec_dir} is not a complete recording")
    depth_scale = float((meta.get("camera") or {})
                        .get("depth_scale_m_per_unit", 0.0001))
    intr = get_intrinsics(meta)
    scale_out = 1000.0 if units == "mm" else 1.0

    if label:
        rows = [r for r in rows if r["label"] == label]

    if expected_distance_m is None:
        probe = [imread_u16(rec_dir / r["filename"]) for r in rows[:8]]
        probe = [p.astype(np.float32) * depth_scale for p in probe
                 if p is not None]
        centre, _ = suggest_distance(np.concatenate(probe)) if probe \
            else (None, 0)
        expected_distance_m = (centre / 1000.0) if centre else 0.15
    lo_m = expected_distance_m - face_depth_mm / 1000.0
    hi_m = expected_distance_m + face_depth_mm / 1000.0

    for r in rows[::max(1, every)]:
        img = imread_u16(rec_dir / r["filename"])
        if img is None:
            continue
        d = img.astype(np.float32) * depth_scale
        d = np.where((d >= lo_m) & (d <= hi_m), d, np.nan)
        if roi:
            x, y, s = roi
            m = np.zeros(d.shape, bool)
            m[y:y + s, x:x + s] = True
            d = np.where(m, d, np.nan)
        if largest_only:
            d, _ = keep_largest_surface(d)
        yield r["frame_index"], r["label"], deproject(d, intr, scale_out)


def newest_recording(root):
    dirs = [d for d in Path(root).iterdir()
            if d.is_dir() and (d / "metadata.json").exists()] \
        if Path(root).exists() else []
    if not dirs:
        sys.exit(f"no recordings found in {root}")
    return max(dirs, key=lambda d: d.stat().st_mtime)


def select_rows(rows, args):
    """Which frames to export, honouring --frame / --range / --label / --trial."""
    sel = rows
    if args.label:
        sel = [r for r in sel if r["label"] == args.label]
        if not sel:
            labels = sorted({r["label"] for r in rows})
            sys.exit(f"no frames labelled {args.label!r}; this recording has "
                     f"{labels}")
    if args.trial is not None:
        sel = [r for r in sel if r["trial_id"] == args.trial]
        if not sel:
            sys.exit(f"no frames in trial {args.trial}")
    if args.frame is not None:
        sel = [r for r in sel if r["frame_index"] == args.frame]
        if not sel:
            sys.exit(f"frame {args.frame} is not in this recording")
    elif args.range:
        try:
            a, b = (int(v) for v in args.range.split(":"))
        except ValueError:
            sys.exit("--range must be START:END")
        sel = [r for r in sel if a <= r["frame_index"] < b]
        if not sel:
            sys.exit(f"no frames in range {args.range}")
    return sel


def face_roi(rec_dir, rows, depth_scale, args):
    """Locate the face with the same fitter the training pipeline uses, so the
    exported model covers exactly the region the model sees.
    Returns (roi, reference_mm) or (None, None)."""
    ref = [r for r in rows if r["label"] == "neutral"] or rows
    frames = []
    for r in ref[:40]:
        img = imread_u16(rec_dir / r["filename"])
        if img is not None:
            frames.append(img)
    if len(frames) < 3:
        return None, None
    over = {"expected_distance_m": args.expected_distance}
    if args.roi:
        over["roi_override"] = [int(v) for v in args.roi.split(",")]
    try:
        pre = DepthPreprocessor(depth_scale, **over).fit(frames)
        return pre.roi, pre.reference_mm
    except ValueError as e:
        print(f"note: could not locate the face automatically ({e})\n"
              f"      exporting the full frame; use --roi X,Y,SIDE to crop.")
        return None, None


def main():
    ap = argparse.ArgumentParser(
        description="Export recorded depth as 3D point clouds / meshes")
    ap.add_argument("--recordings", default=str(HERE / "recordings"))
    ap.add_argument("--recording", default=None,
                    help="a specific recording folder (default: the newest)")
    ap.add_argument("--out", default=None, help="output file")
    ap.add_argument("--out-dir", default=None, help="output dir for --sequence")
    ap.add_argument("--format", choices=["ply", "obj"], default="ply")
    ap.add_argument("--ascii", action="store_true", help="ASCII PLY, not binary")

    ap.add_argument("--frame", type=int, default=None, help="one frame index")
    ap.add_argument("--range", default=None, metavar="START:END")
    ap.add_argument("--label", default=None,
                    help="only frames with this label (e.g. neutral)")
    ap.add_argument("--trial", type=int, default=None)
    ap.add_argument("--sequence", action="store_true",
                    help="one file per frame (4D: shape over time)")
    ap.add_argument("--every", type=int, default=1,
                    help="with --sequence, export every Nth frame")
    ap.add_argument("--average", action="store_true",
                    help="temporal median of consecutive frames into one "
                         "low-noise model (assumes the subject holds still)")
    ap.add_argument("--average-frames", type=int, default=30,
                    help="how many consecutive frames to average (default 30 = "
                         "1 s at 30 fps). Longer windows cut more noise but "
                         "require the subject to be more rigidly still.")
    ap.add_argument("--max-spread-mm", type=float, default=None,
                    help="per-pixel depth variation tolerated before a pixel is "
                         "called 'moving' (default: 5%% of the working "
                         "distance, which stays above the sensor's own noise)")

    ap.add_argument("--mesh", action="store_true", help="triangulated surface")
    ap.add_argument("--points", action="store_true", help="point cloud only")
    ap.add_argument("--max-edge-mm", type=float, default=4.0,
                    help="longest mesh edge; prevents stitching across gaps")
    ap.add_argument("--no-normals", action="store_true")
    ap.add_argument("--no-color", action="store_true",
                    help="omit depth pseudo-colour")
    ap.add_argument("--units", choices=["mm", "m"], default="mm")
    ap.add_argument("--preview", default=None, metavar="PNG",
                    help="also write a shaded relief image, to check the shape "
                         "without opening a 3D viewer")
    ap.add_argument("--center", action="store_true",
                    help="move the origin to the surface centroid")
    ap.add_argument("--decimate", type=int, default=1,
                    help="keep every Nth pixel in each axis (smaller files)")

    ap.add_argument("--full-frame", action="store_true",
                    help="do not crop to the face")
    ap.add_argument("--roi", default=None, metavar="X,Y,SIDE",
                    help="explicit crop box in raw pixels")
    ap.add_argument("--expected-distance", type=float,
                    default=DEFAULT_PARAMS["expected_distance_m"],
                    help="subject distance in metres, for face finding")
    ap.add_argument("--auto-distance", action="store_true",
                    help="measure the subject's distance from the data instead "
                         "of assuming --expected-distance")
    ap.add_argument("--face-depth-mm", type=float, default=80.0,
                    help="keep depth within this many mm of the fitted face "
                         "surface; drops the background behind the subject")
    ap.add_argument("--depth-range", default=None, metavar="MIN,MAX",
                    help="explicit depth band in metres, e.g. 0.10,0.20")
    ap.add_argument("--keep-all", action="store_true",
                    help="keep every surface in the crop, not just the largest "
                         "connected one (default keeps only the face)")
    ap.add_argument("--intrinsics", default=None, metavar="FX,FY,PPX,PPY",
                    help="override the intrinsics (needed for mock/synthetic "
                         "recordings, which have none)")
    args = ap.parse_args()

    rec_dir = Path(args.recording) if args.recording \
        else newest_recording(args.recordings)
    meta, rows = read_session(rec_dir)
    if meta is None or not rows:
        sys.exit(f"{rec_dir} is not a complete recording")
    depth_scale = float((meta.get("camera") or {})
                        .get("depth_scale_m_per_unit", 0.0001))
    intr = get_intrinsics(meta, args.intrinsics)
    scale_out = 1000.0 if args.units == "mm" else 1.0
    unit = args.units
    want_mesh = args.mesh or not args.points

    print(f"recording : {rec_dir.name}")
    print(f"frames    : {len(rows)}   depth unit {depth_scale*1000:.3g} mm")
    print(f"intrinsics: fx={intr[0]:.1f} fy={intr[1]:.1f} "
          f"ppx={intr[2]:.1f} ppy={intr[3]:.1f}")

    sel = select_rows(rows, args)
    print(f"selected  : {len(sel)} frame(s)"
          + (f" labelled {args.label}" if args.label else ""))

    if args.auto_distance:
        probe = []
        for r in sel[:12]:
            img = imread_u16(rec_dir / r["filename"])
            if img is not None:
                probe.append(img.astype(np.float32) * depth_scale)
        centre, covered = suggest_distance(np.concatenate(probe)) \
            if probe else (None, 0.0)
        if centre:
            args.expected_distance = centre / 1000.0
            print(f"auto      : subject measured at ~{centre:.0f} mm "
                  f"({covered*100:.0f}% of near depth within 80 mm)")
        else:
            print("auto      : could not measure a subject distance; "
                  "keeping --expected-distance")

    roi, ref_mm = (None, None) if args.full_frame \
        else face_roi(rec_dir, rows, depth_scale, args)
    if roi:
        print(f"face crop : x={roi[0]} y={roi[1]} side={roi[2]} px, "
              f"surface at {ref_mm:.0f} mm")

    # Depth band. Without this, the corners of a square crop still contain the
    # wall behind the subject, and the "face" model comes out half a metre deep.
    if args.depth_range:
        lo_m, hi_m = (float(v) for v in args.depth_range.split(","))
    else:
        centre_m = (ref_mm / 1000.0) if ref_mm else args.expected_distance
        half = args.face_depth_mm / 1000.0
        lo_m, hi_m = centre_m - half, centre_m + half
    print(f"depth band: {lo_m*1000:.0f} - {hi_m*1000:.0f} mm "
          f"(everything outside is dropped)")

    def crop_mask(shape):
        if roi is None:
            return None
        m = np.zeros(shape, bool)
        x, y, s = roi
        m[y:y + s, x:x + s] = True
        return m

    def emit(depth_m, out_path, note):
        raw_depth = depth_m
        depth_m = np.where((depth_m >= lo_m) & (depth_m <= hi_m),
                           depth_m, np.nan)
        m = crop_mask(depth_m.shape)
        if m is not None:
            depth_m = np.where(m, depth_m, np.nan)
        if not args.keep_all:
            depth_m, n_other = keep_largest_surface(depth_m)
            if n_other > 1:
                print(f"  (kept the largest of {n_other} separate surfaces; "
                      f"--keep-all to export them all)")
        step = max(1, args.decimate)
        if step > 1:
            depth_m = depth_m[::step, ::step]
        pts = deproject(depth_m, intr, scale_out, step=step)

        comments = [f"from {rec_dir.name}", note,
                    f"units {unit}; camera axes +X right +Y down +Z away",
                    f"depth_scale {depth_scale} m/unit"]
        if want_mesh:
            verts, faces = build_mesh(pts, args.max_edge_mm
                                      * (1.0 if unit == "mm" else 0.001))
            normals = None if args.no_normals else vertex_normals(verts, faces)
        else:
            verts = pts[np.isfinite(pts).all(-1)].astype(np.float32)
            faces, normals = None, None
        if not len(verts):
            # "No valid points" alone is useless. Say WHY, with numbers.
            centre, covered = suggest_distance(raw_depth)
            print(f"  ! nothing written: no usable depth survived the "
                  f"{lo_m*1000:.0f}-{hi_m*1000:.0f} mm band.")
            if centre:
                print(f"    The camera actually saw a surface at "
                      f"~{centre:.0f} mm ({covered*100:.0f}% of near depth "
                      f"lies within 80 mm of it), not the "
                      f"{args.expected_distance*1000:.0f} mm this export "
                      f"assumed.\n"
                      f"    Retry with:  --expected-distance "
                      f"{centre/1000:.2f}      (or add --auto-distance)")
            else:
                d = np.asarray(raw_depth) * 1000.0
                v = d[np.isfinite(d) & (d > 0)]
                if v.size:
                    print(f"    Depth in view spans {v.min():.0f}-"
                          f"{v.max():.0f} mm (median {np.median(v):.0f}). "
                          f"Set --expected-distance to your subject's "
                          f"distance in metres.")
                else:
                    print(f"    This frame contains no depth measurements at "
                          f"all -- check the camera and the recording.")
            return None
        if args.center:
            verts = verts - verts.mean(axis=0)
        colors = None if args.no_color else depth_colors(verts)

        if args.format == "obj":
            p = write_obj(out_path, verts, faces, normals, comments)
        else:
            p = write_ply(out_path, verts, faces, normals, colors,
                          binary=not args.ascii, comments=comments)
        ext = verts.max(axis=0) - verts.min(axis=0)
        print(f"  {p.name}: {len(verts)} verts"
              + (f", {len(faces)} tris" if faces is not None else "")
              + f", extent {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} {unit}")
        if args.preview:
            prev = (Path(args.preview) if not args.sequence
                    else p.with_suffix(".png"))
            shade_preview(pts, prev)
            print(f"  preview -> {prev}")
        return p

    if args.sequence:
        out_dir = (Path(args.out_dir).resolve() if args.out_dir
                   else HERE / "exports" / f"{rec_dir.name}_3d")
        chosen = sel[::max(1, args.every)]
        print(f"writing {len(chosen)} file(s) to {out_dir}/")
        for r in chosen:
            img = imread_u16(rec_dir / r["filename"])
            if img is None:
                continue
            depth_m = img.astype(np.float32) * depth_scale
            emit(depth_m, out_dir / f"frame_{r['frame_index']:06d}."
                 f"{args.format}", f"frame {r['frame_index']} "
                                   f"label={r['label']}")
        print(f"\ndone. Open the folder as an image sequence in MeshLab, or "
              f"import into Blender for a 4D playback.")
        return

    if args.average or (args.frame is None and len(sel) > 1):
        # Consecutive frames, capped: averaging assumes the subject held still
        # for the window, and that assumption gets worse the longer it is.
        window = sel[:max(2, args.average_frames)]
        frames = []
        for r in window:
            img = imread_u16(rec_dir / r["filename"])
            if img is not None:
                frames.append(img)
        if not frames:
            sys.exit("no readable frames selected")
        tol = spread_tolerance_mm(ref_mm or args.expected_distance * 1000,
                                  args.max_spread_mm)
        depth_m, n, dropped = average_depth(frames, depth_scale,
                                            max_spread_mm=tol)
        note = (f"temporal median of {n} frames (motion tolerance "
                f"{tol:.1f} mm)" + (f" labelled {args.label}"
                                    if args.label else ""))
        sel_tag = f"avg{n}from{window[0]['frame_index']:06d}"
        print(f"averaging : {n} of {len(sel)} frame(s) "
              f"(noise down ~{np.sqrt(n):.1f}x), tolerance {tol:.1f} mm, "
              f"{dropped*100:.0f}% of pixels rejected as moving")
        if dropped > 0.5:
            print(f"    ! the subject moved more than {tol:.1f} mm at "
                  f"{dropped*100:.0f}% of pixels over these {n} frames, so "
                  f"averaging discarded them.\n"
                  f"      Use a shorter window (--average-frames 10), raise "
                  f"--max-spread-mm, or take a single frame (--frame N) if the "
                  f"subject is not head-fixed.")
    else:
        r = sel[0]
        img = imread_u16(rec_dir / r["filename"])
        if img is None:
            sys.exit(f"could not read {r['filename']}")
        depth_m = img.astype(np.float32) * depth_scale
        note = f"single frame {r['frame_index']} label={r['label']}"
        sel_tag = f"frame{r['frame_index']:06d}"

    # Default output goes to exports/ next to the program, NOT the current
    # working directory, so files are always findable and the raw recordings
    # folder stays untouched.
    # Name says exactly WHICH frames are inside: a single-frame export is one
    # moment out of hundreds, and calling it "all" invites misreading it as the
    # whole recording.
    default_name = (f"{rec_dir.name}_"
                    + (f"{args.label}_" if args.label else "")
                    + f"{sel_tag}_"
                    + f"{'mesh' if want_mesh else 'points'}.{args.format}")
    out = Path(args.out).resolve() if args.out \
        else (HERE / "exports" / default_name)
    written = emit(depth_m, out, note)
    if written:
        print(f"\nsaved to: {written.resolve()}")
        print(f"open it in MeshLab / CloudCompare / Blender, or in Python:\n"
              f"  import trimesh; trimesh.load(r'{written.resolve()}').show()")


if __name__ == "__main__":
    main()
