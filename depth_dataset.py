"""
Dataset access for the built depth dataset.

    from depth_dataset import DepthClipDataset
    train = DepthClipDataset("dataset", split="train", mode="deform",
                             augment=True)
    x, y = train[0]          # x: (2, T, S, S) float32, y: int label id

Channel 0 is relative depth scaled to [-1, 1]; channel 1 is the validity mask.
The final scaling and the deform subtraction both go through
preprocessing.to_model_input, which live_infer.py also calls -- that shared
call is what guarantees training and real-time inference see identical inputs.

Works without torch (returns numpy arrays); DataLoader support activates if
torch is installed.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from preprocessing import (PREPROC_VERSION, DepthPreprocessor, input_scale,
                           to_model_input)

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:                                     # numpy-only fallback
    torch = None

    class _TorchDataset:                                # noqa: D101
        pass


class DepthClipDataset(_TorchDataset):
    """Clips indexed by dataset/manifest_clips.csv, read from per-session
    float16 memmaps (so a dataset larger than RAM is fine)."""

    def __init__(self, root, split="train", mode="absolute", augment=False,
                 labels=None, strict_version=True, aug=None):
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.augment = augment
        # depth_mm randomises the whole-face DC offset so the model cannot key
        # on slow session drift. It is capped at the level build_dataset.py
        # warns about (DRIFT_WARN_MM = 2 mm) plus margin: pushing it much
        # higher starts swamping genuine deformation, which for a face is only
        # millimetres. If the build reports a larger per-label DC spread, fix it
        # at the source with `build_dataset.py --per-frame-dc roi_median`
        # instead of blurring the input here.
        self.aug = {"shift_px": 4, "depth_mm": 3.0, "noise_mm": 0.3,
                    "dropout": 0.05, "time_jitter": 2, "flip": False,
                    **(aug or {})}

        spec_path = self.root / "dataset_spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(
                f"{spec_path} not found -- run build_dataset.py first")
        self.spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if strict_version and self.spec.get("preproc_version") \
                != PREPROC_VERSION:
            raise ValueError(
                f"dataset was built with preprocessing "
                f"{self.spec.get('preproc_version')}, code is "
                f"{PREPROC_VERSION} -- re-run build_dataset.py --rebuild")

        self.params = self.spec["preproc_params"]
        self.out_size = int(self.params["out_size"])
        # Same scale live_infer.py uses, via the same shared function.
        self.clip_mm = input_scale(self.params, self.mode)

        rows = []
        with open(self.root / "manifest_clips.csv", newline="",
                  encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if split in (None, "all") or row["split"] == split:
                    rows.append(row)

        # Label space: keep it fixed across splits/runs so ids never shift.
        if labels is not None:
            self.labels = list(labels)
        else:
            lm = self.spec["label_map"]
            self.labels = [k for k, _ in sorted(lm.items(),
                                                key=lambda kv: kv[1])]
        self.label_to_id = {lab: i for i, lab in enumerate(self.labels)}

        self.clips = [r for r in rows if r["label"] in self.label_to_id]
        if len(self.clips) != len(rows):
            dropped = {r["label"] for r in rows} - set(self.label_to_id)
            print(f"[DepthClipDataset] ignoring clips with unknown labels: "
                  f"{sorted(dropped)}")

        self._arrays = {}      # session -> memmap
        self._refs = {}        # session -> reference image

    # -- lazy per-session resources --
    def _array(self, session):
        arr = self._arrays.get(session)
        if arr is None:
            arr = np.load(self.root / "sessions" / f"{session}.npy",
                          mmap_mode="r")
            self._arrays[session] = arr
        return arr

    def _reference(self, session):
        ref = self._refs.get(session)
        if ref is None:
            info = json.loads((self.root / "sessions" / f"{session}.json")
                              .read_text(encoding="utf-8"))
            pre = DepthPreprocessor.from_state_dict(info["preproc"],
                                                    strict=False)
            ref = pre.reference_image
            self._refs[session] = ref
        return ref

    # -- interface --
    def __len__(self):
        return len(self.clips)

    def class_counts(self):
        c = Counter(r["label"] for r in self.clips)
        return [c.get(lab, 0) for lab in self.labels]

    def class_weights(self):
        """Inverse-frequency weights for imbalanced conditions; classes absent
        from this split get weight 0 so they can't skew the loss."""
        counts = np.asarray(self.class_counts(), np.float64)
        w = np.zeros_like(counts)
        present = counts > 0
        w[present] = counts[present].sum() / (present.sum() * counts[present])
        return w.astype(np.float32)

    def __getitem__(self, i):
        row = self.clips[i]
        session, start = row["session"], int(row["start"])
        length = int(row["length"])
        arr = self._array(session)

        if self.augment and self.aug["time_jitter"]:
            j = np.random.randint(-self.aug["time_jitter"],
                                  self.aug["time_jitter"] + 1)
            start = int(np.clip(start + j, 0, max(0, arr.shape[0] - length)))

        clip = np.asarray(arr[start:start + length], np.float32)
        if clip.shape[0] < length:                      # pad short tail
            pad = np.repeat(clip[-1:], length - clip.shape[0], axis=0)
            clip = np.concatenate([clip, pad], 0)

        ref = self._reference(session) if self.mode == "deform" else None
        if self.augment:
            clip, ref = self._augment(clip, ref)

        x = to_model_input(clip, reference_image=ref,
                           clip_mm_max=self.clip_mm, mode=self.mode)
        y = self.label_to_id[row["label"]]
        if torch is not None:
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
        return x, y

    # -- augmentation (applied in millimetre space, before to_model_input) --
    def _augment(self, clip, ref):
        a = self.aug
        if a["shift_px"]:
            dx = np.random.randint(-a["shift_px"], a["shift_px"] + 1)
            dy = np.random.randint(-a["shift_px"], a["shift_px"] + 1)
            clip = np.roll(clip, (dy, dx), axis=(1, 2))
            if ref is not None:
                ref = np.roll(ref, (dy, dx), axis=(0, 1))
        if a["flip"] and np.random.rand() < 0.5:
            clip = clip[:, :, ::-1].copy()
            if ref is not None:
                ref = ref[:, ::-1].copy()
        if a["depth_mm"]:
            # small whole-face distance change: does NOT move the reference,
            # mimicking a genuine shift of the subject relative to baseline
            clip = clip + np.random.uniform(-a["depth_mm"], a["depth_mm"])
        if a["noise_mm"]:
            clip = clip + np.random.normal(0, a["noise_mm"],
                                           clip.shape).astype(np.float32)
        if a["dropout"]:
            holes = np.random.rand(*clip.shape) < a["dropout"]
            clip = np.where(holes, np.nan, clip)        # simulate depth holes
        return clip, ref


def make_loader(dataset, batch_size=16, shuffle=True, num_workers=0):
    """DataLoader with a Windows-friendly default of 0 workers (memmaps and
    spawn-based workers interact badly)."""
    if torch is None:
        raise ImportError("torch is required for make_loader")
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    for split in ("train", "val", "test"):
        try:
            ds = DepthClipDataset(root, split=split)
        except (FileNotFoundError, ValueError) as e:
            sys.exit(str(e))
        print(f"{split:5s}: {len(ds):5d} clips  counts="
              f"{dict(zip(ds.labels, ds.class_counts()))}")
        if len(ds):
            x, y = ds[0]
            shape = tuple(x.shape)
            print(f"        sample x{shape} y={int(y)} "
                  f"range=[{float(x[0].min()):.2f}, {float(x[0].max()):.2f}] "
                  f"valid={float(x[1].mean()):.2f}")
