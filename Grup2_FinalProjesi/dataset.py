"""
dataset.py - HDIN veri seti yukleyicisi
========================================
Gercek klasor yapisi:
    HDIN/
      training/
        steer001/
          images/*.jpg       (timestamp isimli)
          label.txt          (timestamp,value - iki sutun)
        collision001/
          images/*.jpg
          labels.txt         (tek sutun: 0.0/1.0)
"""

import re
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# Sabitler
IMG_H   = 244
IMG_W   = 324
SEQ_LEN = 4
S_MAX   = 40.0   # HDIN maks angular velocity (derece/s)


def parse_trajectory(label_path: str) -> list[dict]:
    """
    Format A - Steering (iki sutun): timestamp,value
    Format B - Collision (tek sutun): 0.0 / 1.0
    """
    entries = []
    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return entries

    # Baslik satirini atla
    if re.match(r"^[a-zA-Z_]", lines[0]):
        lines = lines[1:]
    if not lines:
        return entries

    has_comma = "," in lines[0]

    for line in lines:
        if has_comma:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                ts    = int(parts[0].strip())
                value = float(parts[1].strip())
                entries.append({"timestamp": ts, "value": value,
                                "label_type": "steering"})
            except ValueError:
                continue
        else:
            try:
                value = float(line.strip())
                entries.append({"timestamp": None, "value": value,
                                "label_type": "collision"})
            except ValueError:
                continue
    return entries


def load_folder_samples(folder: Path) -> list[dict]:
    """
    Bir steer* veya collision* klasoründen SEQ_LEN'li dizi ornekleri uretir.
    Steering icin: image timestamp ile label timestamp eslestirmesi yapilir.
    """
    images_dir = folder / "images"

    # Dosya adini belirle
    name = folder.name.lower()
    if name.startswith("steer"):
        label_file = folder / "label.txt"
    else:
        label_file = folder / "labels.txt"
        if not label_file.exists():
            label_file = folder / "label.txt"

    if not images_dir.exists() or not label_file.exists():
        return []

    # Goruntu dosyalarini sirala (jpg ve png destekli)
    img_files = sorted(
        [p for p in images_dir.iterdir()
         if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda p: p.stem,
    )

    labels = parse_trajectory(str(label_file))

    if not img_files or not labels:
        return []

    # Steering icin timestamp eslestirmesi
    if labels[0]["label_type"] == "steering":
        # Her image icin en yakin timestamp'li label'i bul
        label_ts = np.array([l["timestamp"] for l in labels])
        label_vals = np.array([l["value"] for l in labels])

        matched = []
        for img in img_files:
            try:
                img_ts = int(img.stem)
            except ValueError:
                continue
            idx = np.argmin(np.abs(label_ts - img_ts))
            matched.append({
                "img_path": str(img),
                "value": float(np.clip(label_vals[idx] / S_MAX, -1.0, 1.0)),
                "label_type": "steering",
            })

        if len(matched) < SEQ_LEN:
            return []

        samples = []
        for i in range(SEQ_LEN - 1, len(matched)):
            seq_paths = [matched[i - SEQ_LEN + 1 + k]["img_path"]
                         for k in range(SEQ_LEN)]
            samples.append({
                "image_paths": seq_paths,
                "label":       matched[i]["value"],
                "label_type":  "steering",
            })
        return samples

    else:
        # Collision: dogrudan sirali eslestirme
        n = min(len(img_files), len(labels))
        if n < SEQ_LEN:
            return []

        samples = []
        for i in range(SEQ_LEN - 1, n):
            seq_paths = [str(img_files[i - SEQ_LEN + 1 + k])
                         for k in range(SEQ_LEN)]
            value = float(labels[i]["value"])
            samples.append({
                "image_paths": seq_paths,
                "label":       value,
                "label_type":  "collision",
            })
        return samples


class HDINDataset(Dataset):
    def __init__(self, root_dir: str, augment: bool = False,
                 img_size: tuple = (IMG_H, IMG_W)):
        self.root_dir = Path(root_dir)
        self.augment  = augment
        self.img_size = img_size
        self.samples: list[dict] = []
        self._scan()

    def _scan(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dizin bulunamadi: {self.root_dir}")
        steer_count = 0
        collision_count = 0
        for folder in sorted(self.root_dir.iterdir()):
            if not folder.is_dir():
                continue
            name = folder.name.lower()
            if name.startswith("steer") or name.startswith("collision"):
                before = len(self.samples)
                self.samples.extend(load_folder_samples(folder))
                added = len(self.samples) - before
                if name.startswith("steer"):
                    steer_count += added
                else:
                    collision_count += added
        print(f"  [{self.root_dir.name}] steering:{steer_count}  "
              f"collision:{collision_count}  toplam:{len(self.samples)}")
        if not self.samples:
            raise RuntimeError(
                f"Hic ornek bulunamadi: {self.root_dir}"
            )

    def _load(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        return TF.to_tensor(img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s          = self.samples[idx]
        label      = s["label"]
        label_type = s["label_type"]
        frames     = [self._load(p) for p in s["image_paths"]]

        if self.augment and random.random() > 0.5:
            frames = [TF.hflip(f) for f in frames]
            if label_type == "steering":
                label = -label

        return {
            "images":     torch.stack(frames),
            "label":      torch.tensor(label, dtype=torch.float32),
            "label_type": label_type,
        }


def collate_multitask(batch: list[dict]) -> dict:
    si, sl, ci, cl = [], [], [], []
    for item in batch:
        if item["label_type"] == "steering":
            si.append(item["images"]); sl.append(item["label"])
        else:
            ci.append(item["images"]); cl.append(item["label"])
    return {
        "steer_images":     torch.stack(si) if si else None,
        "steer_labels":     torch.stack(sl) if sl else None,
        "collision_images": torch.stack(ci) if ci else None,
        "collision_labels": torch.stack(cl) if cl else None,
    }


def build_dataloader(root_dir: str, batch_size: int = 8,
                     augment: bool = False, shuffle: bool = True,
                     num_workers: int = 4,
                     img_size: tuple = (IMG_H, IMG_W)) -> DataLoader:
    ds = HDINDataset(root_dir=root_dir, augment=augment, img_size=img_size)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_multitask,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )


def diagnose(root_dir: str):
    root = Path(root_dir)
    print(f"\n=== TESHIS: {root_dir} ===")
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        if not (name.startswith("steer") or name.startswith("collision")):
            continue
        images_dir = folder / "images"
        exts = set(p.suffix.lower() for p in images_dir.iterdir()) \
               if images_dir.exists() else set()
        img_count = len([p for p in images_dir.iterdir()
                         if p.suffix.lower() in (".jpg",".jpeg",".png")]) \
                    if images_dir.exists() else 0

        lf = (folder/"label.txt") if name.startswith("steer") else (folder/"labels.txt")
        if not lf.exists():
            lf = folder/"label.txt"
        if lf.exists():
            labels = parse_trajectory(str(lf))
            fmt = labels[0]["label_type"] if labels else "bos"
            print(f"  {folder.name}: {img_count} img {exts}, "
                  f"{len(labels)} label, format={fmt}")
        else:
            print(f"  {folder.name}: label dosyasi YOK")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "HDIN/training"
    diagnose(root)
    for split in ["training", "validation", "testing"]:
        path = str(Path(root).parent / split)
        if Path(path).exists():
            diagnose(path)