from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


def _find_single(path: Path, pattern: str) -> Path:
    """Find exactly one match under path; raise if none."""
    matches = list(path.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find '{pattern}' under {path}")
    return matches[0]


def load_devkit_meta(devkit_root: Path) -> dict[str, int]:
    """
    Build wordnet_id (WNID) -> 0-based class index from devkit meta.mat.
    Train dirs are named by WNID; we need WNID -> index, not ILSVRC2012_ID.
    """
    meta_path = _find_single(Path(devkit_root), "meta.mat")
    meta = loadmat(str(meta_path), squeeze_me=True, struct_as_record=False)
    synsets = meta["synsets"]

    # Get WNID and ILSVRC2012_ID for each synset (may be 1000 + 860 high-level)
    if hasattr(synsets, "WNID"):
        wnids = np.atleast_1d(synsets.WNID)
        ids = np.atleast_1d(synsets.ILSVRC2012_ID)
    else:
        wnids = np.array([s.WNID for s in np.atleast_1d(synsets)])
        ids = np.array([s.ILSVRC2012_ID for s in np.atleast_1d(synsets)])

    # Only low-level synsets (ILSVRC2012_ID 1..1000); map WNID -> 0-based index
    out = {}
    for i, (wid, ilsvrc_id) in enumerate(zip(wnids, ids)):
        if 1 <= int(ilsvrc_id) <= 1000:
            out[str(wid).strip()] = int(ilsvrc_id) - 1
    return out


def load_val_ground_truth(devkit_root: Path) -> list[int]:
    """
    Validation labels (1-based in file -> 0-based here).

    Searches devkit_root recursively for ILSVRC2012_validation_ground_truth.txt.
    """
    gt_path = _find_single(
        devkit_root, "ILSVRC2012_validation_ground_truth.txt"
    )
    with open(gt_path) as f:
        return [int(line.strip()) - 1 for line in f]


def collect_train_samples(
    imagenet_root: Path, wn_to_idx: dict[str, int]
) -> list[tuple[str, int]]:
    """List (absolute_path, class_index) for all training images."""
    train_dir = imagenet_root / "train"
    out: list[tuple[str, int]] = []
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        wn_id = class_dir.name
        cls = wn_to_idx.get(wn_id)
        if cls is None:
            continue
        for f in sorted(class_dir.glob("*.JPEG")):
            out.append((str(f.resolve()), cls))
    return out


def collect_val_samples(
    imagenet_root: Path, gt: list[int]
) -> list[tuple[str, int]]:
    """List (absolute_path, class_index) for validation images."""
    val_dir = imagenet_root / "val"
    out: list[tuple[str, int]] = []
    for i, cls in enumerate(gt):
        name = f"ILSVRC2012_val_{i + 1:08d}.JPEG"
        path = val_dir / name
        if path.exists():
            out.append((str(path.resolve()), cls))
    return out


def collect_test_samples(imagenet_root: Path) -> list[tuple[str, int]]:
    """
    List (absolute_path, dummy_class_index) for test images.

    Test set has no labels; we use -1 as a placeholder class_id.
    Assumes files like ILSVRC2012_test_00000001.JPEG, ...
    """
    test_dir = imagenet_root / "test"
    out: list[tuple[str, int]] = []
    files = sorted(test_dir.glob("ILSVRC2012_test_*.JPEG"))
    for f in files:
        out.append((str(f.resolve()), -1))
    return out


def write_shard(
    samples: list[tuple[str, int]],
    shard_path: str,
    shard_idx: int,
    reencode_jpeg: bool = False,
) -> list[dict]:
    """Write one .tar shard; return index entries."""
    index: list[dict] = []
    with tarfile.open(shard_path, "w:") as tar:
        for i, (img_path, class_id) in enumerate(samples):
            key = f"{shard_idx:06d}_{i:08d}"

            # Add image
            if reencode_jpeg:
                with Image.open(img_path) as im:
                    im.load()
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, "JPEG", quality=95)
                    data = buf.getvalue()
                buf = io.BytesIO(data)
                ti = tarfile.TarInfo(name=f"{key}.jpg")
                ti.size = len(data)
                tar.addfile(ti, buf)
            else:
                tar.add(img_path, arcname=f"{key}.jpg")

            # Add label as a small .cls text file (0-based class index, -1 for test)
            cls_bytes = f"{class_id}\n".encode()
            cls_info = tarfile.TarInfo(name=f"{key}.cls")
            cls_info.size = len(cls_bytes)
            tar.addfile(cls_info, io.BytesIO(cls_bytes))

            index.append(
                {
                    "key": key,
                    "shard": os.path.basename(shard_path),
                    "class_id": class_id,
                }
            )
    return index


def build_webdataset(
    imagenet_root: Path,
    output_dir: Path,
    samples_per_shard: int = 1000,
    split: str = "train",
    reencode_jpeg: bool = False,
) -> list[dict]:
    """
    Build WebDataset shards + JSON index for ImageNet.

    imagenet_root should contain:
      train/  (nXXXXX folders)          for split='train'
      val/    (ILSVRC2012_val_*.JPEG)   for split='val'
      test/   (ILSVRC2012_test_*.JPEG)  for split='test'
      devkit_t12/ (we search inside it for meta.mat and val gt)
    """
    imagenet_root = Path(imagenet_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    devkit_root = imagenet_root / "devkit_t12"

    if split == "train":
        wn_to_idx = load_devkit_meta(devkit_root)
        samples = collect_train_samples(imagenet_root, wn_to_idx)
    elif split == "val":
        gt = load_val_ground_truth(devkit_root)
        samples = collect_val_samples(imagenet_root, gt)
    elif split == "test":
        samples = collect_test_samples(imagenet_root)
    else:
        raise ValueError(f"Unsupported split: {split!r}")

    print(f"Found {len(samples)} samples for split='{split}'")

    all_index: list[dict] = []
    for start in range(0, len(samples), samples_per_shard):
        shard_idx = start // samples_per_shard
        shard_path = output_dir / f"imagenet-{split}-{shard_idx:06d}.tar"
        chunk = samples[start : start + samples_per_shard]
        all_index.extend(
            write_shard(
                chunk, str(shard_path), shard_idx, reencode_jpeg=reencode_jpeg
            )
        )

    index_path = output_dir / f"imagenet-{split}.index.json"
    with open(index_path, "w") as f:
        json.dump(all_index, f, indent=0)

    return all_index


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert ImageNet to indexed WebDataset"
    )
    p.add_argument(
        "imagenet_root",
        type=Path,
        help="Root of extracted ILSVRC2012 (train/, val/, test/, devkit_t12/)",
    )
    p.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for .tar shards and index",
    )
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--samples-per-shard", type=int, default=1000)
    p.add_argument(
        "--reencode",
        action="store_true",
        help="Decode and re-encode images as JPEG",
    )
    args = p.parse_args()

    build_webdataset(
        args.imagenet_root,
        args.output_dir,
        samples_per_shard=args.samples_per_shard,
        split=args.split,
        reencode_jpeg=args.reencode,
    )
    print(f"Done. Wrote shards and index to {args.output_dir}")


if __name__ == "__main__":
    main()