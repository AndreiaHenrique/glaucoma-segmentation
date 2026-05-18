"""
Inference-only script: generate predicted cup/disc masks for a folder of
unlabeled fundus images using the UPerNet 5-fold ensemble trained in
`upernet_optuna.py`.

Usage (from inside src/):
    python predict_masks.py

Configuration via environment variables:
    NEW_IMG_DIR        default: ./data/new_images
    PRED_OUT_DIR       default: ./data/predicted_masks
    UPERNET_MODEL_DIR  default: ./checkpoints/upernet_models
    SAVE_OVERLAY       "1" to also save a coloured overlay alongside each mask
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from skimage.io import imread, imsave
import matplotlib.pyplot as plt

from pipeline import (
    IMG_SIZE, IMG_EXTS, IMAGENET_MEAN, IMAGENET_STD, device,
)
from upernet_optuna import build_model


NEW_IMG_PATH = os.environ.get("NEW_IMG_DIR",   "./data/new_images")
OUT_DIR      = os.environ.get("PRED_OUT_DIR",  "./data/predicted_masks")
MODEL_DIR    = os.environ.get("UPERNET_MODEL_DIR", "./checkpoints/upernet_models")
SAVE_OVERLAY = os.environ.get("SAVE_OVERLAY", "0") == "1"


def list_images(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    )


transform = A.Compose([
    A.Resize(*IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
    ToTensorV2(),
])


class InferenceDataset(Dataset):
    def __init__(self, images_list, transform=None):
        self.images_list = images_list
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        path = self.images_list[idx]
        img = imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        img = img.astype(np.uint8)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img.float(), path


@torch.no_grad()
def predict_ensemble(model_paths, loader):
    sum_probs = None
    for path in model_paths:
        model = build_model()
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        probs_all = []
        for imgs, _ in loader:
            imgs = imgs.to(device)
            probs_all.append(torch.softmax(model(imgs), dim=1).cpu().numpy())
        probs_all = np.concatenate(probs_all, axis=0)
        sum_probs = probs_all if sum_probs is None else sum_probs + probs_all
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.argmax(sum_probs / len(model_paths), axis=1)


def main():
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    image_files = list_images(NEW_IMG_PATH)
    print(f"Images: {len(image_files)}")

    model_paths = sorted(
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR) if f.endswith(".pth")
    )
    print(f"Ensemble checkpoints: {len(model_paths)}")
    assert model_paths, f"No .pth files found in {MODEL_DIR}"

    loader = DataLoader(InferenceDataset(image_files, transform),
                        batch_size=1, shuffle=False)

    print("Generating masks...")
    pred_masks = predict_ensemble(model_paths, loader)

    # ---- Save proper segmentation masks (1-channel, classes 0/1/2) ----
    # We scale to {0, 127, 255} so the file is also human-viewable.
    LUT = np.array([0, 127, 255], dtype=np.uint8)   # 0=bg, 1=cup, 2=disc
    for path, mask in zip(image_files, pred_masks):
        stem = os.path.splitext(os.path.basename(path))[0]
        out_mask = LUT[mask.astype(np.uint8)]
        imsave(os.path.join(OUT_DIR, f"{stem}_mask.png"), out_mask)

        if SAVE_OVERLAY:
            plt.imsave(os.path.join(OUT_DIR, f"{stem}_overlay.png"),
                       mask.astype(np.uint8), cmap="jet")

    print(f"Done. Wrote {len(pred_masks)} masks to {OUT_DIR}")


if __name__ == "__main__":
    main()