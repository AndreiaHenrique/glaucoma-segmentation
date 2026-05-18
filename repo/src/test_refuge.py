"""
External (zero-shot) validation on the REFUGE dataset using the UPerNet
5-fold ensemble, with an "Oracle Localization" crop derived from the
ground-truth masks (matches the paper's protocol).

Mask conventions for REFUGE are different from the training dataset
(white = background, grey = disc, black = cup), so we keep a dedicated
Dataset class here. All other utilities (build_model, metrics, ensemble)
are imported from the shared modules.

Configuration:
    REFUGE_IMG_DIR     default: ./data/REFUGE/test/images
    REFUGE_MASK_DIR    default: ./data/REFUGE/test/mask
    UPERNET_MODEL_DIR  default: ./checkpoints/upernet_models
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from skimage.io import imread
import matplotlib.pyplot as plt

from pipeline import (
    NUM_CLASSES, device, IMG_SIZE, IMG_EXTS,
    IMAGENET_MEAN, IMAGENET_STD,
    ensemble_predict, collect_test_arrays, report_metrics,
    visualise_best_worst,
)
from upernet_optuna import build_model  # the best model from the paper


REFUGE_IMG_PATH  = os.environ.get("REFUGE_IMG_DIR",  "./data/REFUGE/test/images")
REFUGE_MASK_PATH = os.environ.get("REFUGE_MASK_DIR", "./data/REFUGE/test/mask")
MODEL_DIR        = os.environ.get("UPERNET_MODEL_DIR",
                                  "./checkpoints/upernet_models")


def list_pairs(img_dir, mask_dir):
    imgs = {os.path.splitext(f)[0]: os.path.join(img_dir, f)
            for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)}
    masks = {os.path.splitext(f)[0]: os.path.join(mask_dir, f)
             for f in os.listdir(mask_dir) if f.lower().endswith(IMG_EXTS)}
    common = sorted(set(imgs) & set(masks))
    print(f"REFUGE matched pairs: {len(common)}")
    return [imgs[k] for k in common], [masks[k] for k in common]


def discretize_refuge_mask(mask):
    """REFUGE: white -> background (0), black -> cup (1), grey -> disc (2)."""
    m = mask.astype(np.float32)
    if m.max() <= 1.5:
        m = m * 255.0
    out = np.zeros_like(m, dtype=np.uint8)
    out[m > 200] = 0
    out[m < 50]  = 1
    out[(m >= 50) & (m <= 200)] = 2
    return out


def get_bounding_box(mask, pad=100):
    """Square bbox around everything that is not background (Oracle crop)."""
    rows = np.any(mask < 200, axis=1)
    cols = np.any(mask < 200, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0, mask.shape[0], 0, mask.shape[1]
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    ymin = max(0, ymin - pad); ymax = min(mask.shape[0], ymax + pad)
    xmin = max(0, xmin - pad); xmax = min(mask.shape[1], xmax + pad)
    h, w = ymax - ymin, xmax - xmin
    size = max(h, w)
    cy, cx = ymin + h // 2, xmin + w // 2
    y0 = max(0, cy - size // 2); y1 = min(mask.shape[0], cy + size // 2)
    x0 = max(0, cx - size // 2); x1 = min(mask.shape[1], cx + size // 2)
    return y0, y1, x0, x1


class RefugeDataset(Dataset):
    def __init__(self, images_list, masks_list, transform=None):
        self.images_list = images_list
        self.masks_list  = masks_list
        self.transform   = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img = imread(self.images_list[idx])
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        img = img.astype(np.uint8)
        mask_orig = imread(self.masks_list[idx], as_gray=True)

        y0, y1, x0, x1 = get_bounding_box(mask_orig, pad=150)
        img_c  = img[y0:y1, x0:x1]
        mask_c = mask_orig[y0:y1, x0:x1]
        mask_final = discretize_refuge_mask(mask_c)

        if self.transform:
            t = self.transform(image=img_c, mask=mask_final)
            img_c, mask_final = t["image"], t["mask"]
        return img_c.float(), mask_final.long()


def main():
    print(f"Device: {device}")
    refuge_imgs, refuge_masks = list_pairs(REFUGE_IMG_PATH, REFUGE_MASK_PATH)

    test_transform = A.Compose([
        A.Resize(*IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])

    test_ds = RefugeDataset(refuge_imgs, refuge_masks, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=(device == "cuda"))

    fold_paths = sorted([os.path.join(MODEL_DIR, f)
                         for f in os.listdir(MODEL_DIR) if f.endswith(".pth")])
    print(f"Ensemble checkpoints: {len(fold_paths)}")

    imgs_test, masks_test = collect_test_arrays(test_loader)
    preds_test = ensemble_predict(build_model, fold_paths, test_loader)
    print("Ensemble predictions:", preds_test.shape)

    print("\n" + "=" * 40)
    print(" REFUGE results (Oracle Crop)")
    print("=" * 40)
    report_metrics(masks_test, preds_test)
    visualise_best_worst(imgs_test, masks_test, preds_test, k=3)


if __name__ == "__main__":
    main()
