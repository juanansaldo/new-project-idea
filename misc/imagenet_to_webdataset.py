from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from pathlib import Path

from PIL import Image
from scipy.io import loadmat


def load_devkit_meta(devkit_path: Path) -> dict[str, int]:
    """Build wordnet_id -> 0-based class index from devkit meta.mat."""
    meta_path = devkit_path / "data" / "meta.mat"
    meta = loadmat(str(meta_path), squeeze_me=True, struct_as_record=False)
    synsets = meta["synsets"]
    if hasattr(synsets, "ILSVRC2012_ID"):
        ids = synsets.ILSVRC2012_ID
    else:
        ids = [s.ILSVRC2012_ID for s in synsets]
    return {str(wid): i for i, wid in enumerate(ids)}


def load_val_ground_truth(devkit_path: Path) -> list[int]:
    """Validation labels (1-based in file -> 0-based here)."""
    gt_path = devkit_path / "data" / "ILSVRC2012_validation_ground_truth.txt"
    with open(gt_path) as f:
        return [int(line.strip()) - 1 for line in f]


def collect_train_samples(
    imagenet_root: Path, wn_to_idx: dict[str, int]
) -> list[tuple[str, int]]:
    """List (absolute_path, class_index) for all training images."""
    train_dir = imagenet_root / "train"
    out = []
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
    out = []
    for i in range(len(gt)):
        name = f"ILSVRC2012_val_{i + 1:08d}.JPEG"
        path = val_dir / name
        if path.exists():
            out.append((str(path.resolve()), gt[i]))
    return out


def write_shard(
    samples: list[tuple[str, int]],
    shard_path: str,
    shard_idx: int,
    reencode_jpeg: bool = False,
) -> list[dict]:
    """Write one .tar shard; return index entries."""
    index = []
    with tarfile.open(shard_path, "w:") as tar:
        for i, (img_path, class_id) in enumerate(samples):
            key = f"{shard_idx:06d}_{i:08d}"
            if reencode_jpeg:
                with Image.open(img_path) as im:
                    im.load()
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, "JPEG", quality=95)
                    buf.seek(0)
                    ti = tarfile.TarInfo(name=f"{key}.jpg")
                    ti.size = len(buf.getvalue())
                    tar.addfile(ti, buf)
            else:
                tar.add(img_path, arcname=f"{key.jpg}")
            cls_data = f"{class_id}\n".encode()
            ti = tarfile.TarInfo(name=f"{key}.jpg")
            ti.size = len(cls_data)
            tar.addfile(ti, io.BytesIO(cls_data))
            index.append({
                "key": key,
                "shard": os.path.basename(shard_path),
                "class_id": class_id,
            })
    return index


def build_webdataset(
    imagenet_root: Path,
    output_dir: Path,
    samples_per_shard: int = 1000,
    split: str = "train",
    reencode_jpeg: bool = False,
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    devkit_path = imagenet_root / "devkit"

    if split == "train":
        wn_to_idx = load_devkit_meta(devkit_path)
        samples = collect_train_samples(imagenet_root, gt)

    all_index = []
    for start in range(0, len(samples), samples_per_shard):
        shard_idx = start // samples_per_shard
        shard_path = output_dir / f"imagenet-{split}-{shard_idx:06d}.tar"
        chunk = samples[start:start + samples_per_shard]
        all_index.extend(
            write_shard(chunk, str(shard_path), shard_idx, reencode_jpeg=reencode_jpeg)
        )
    
    index_path = output_dir / f"imagenet-{split}.index.json"
    with open(index_path, "w") as f:
        json.dump(all_index, f, indent=0)
    return all_index


def main():
    p = argparse.ArgumentParser(
        description="Convert ImageNet to indexed WebDataset"
    )
    p.add_argument(
        "imagenet_root",
        type=Path,
        help="Root of extracted ILSVRC2012 (train/, val/, devkit/)",
    )
    p.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for .tar shards and index",
    )
    p.add_argument("--split", choices=("train", "val"), default="train")
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