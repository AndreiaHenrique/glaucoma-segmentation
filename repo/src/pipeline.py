"""
Shared training / evaluation pipeline for the three glaucoma-segmentation
architectures compared in the paper.

By design, this module contains EVERYTHING that is common to UPerNet,
MAnet and DPT — so the only architecture-specific code lives in the
respective wrapper scripts (`upernet_optuna.py`, `manet_optuna.py`,
`dpt_optuna.py`). That guarantees a fair comparison: same data, same
splits, same augmentation, same loss, same optimiser, same scheduler,
same Optuna search space, same ensemble.

Defaults match the methodology described in the paper:
    * Resolution: 224 x 224
    * Classes: 0 (background) | 1 (cup) | 2 (disc/rim)
    * Augmentation: HFlip, VFlip, Rotate(±15°), Brightness/Contrast
    * Loss: alpha * Dice + (1 - alpha) * WeightedCE
            (alpha tuned by Optuna; class weights by inverse frequency,
             computed per fold from training masks only)
    * Optuna (TPE, seed=42, 10 trials, 3 epochs):
            lr in [1e-5, 1e-3] (log)
            batch_size in {4, 8}
            alpha in [0.3, 0.7]
            Fixed holdout: 20% of the training pool
    * CV: 5-fold (random KFold) on the remaining 80 % of training data
    * Optimiser: Adam
    * Scheduler: ReduceLROnPlateau(factor=0.5, patience=3)
    * Early stopping: patience = 8 epochs on val loss
    * Epochs: 50 per fold
    * Ensemble: average of softmax over the 5 folds, then argmax
"""

from __future__ import annotations

import os
import random
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

from skimage.io import imread
from skimage.transform import resize
from skimage import measure
from skimage.morphology import binary_dilation, disk
from scipy.spatial.distance import directed_hausdorff

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold, train_test_split

import optuna
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# 1.  Reproducibility
# ------------------------------------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
_G = torch.Generator(); _G.manual_seed(SEED)


# ------------------------------------------------------------------
# 2.  Constants
# ------------------------------------------------------------------
IMG_SIZE      = (224, 224)
NUM_CLASSES   = 3
CUP_CLASS     = 1
RIM_CLASS     = 2

EPOCHS        = int(os.environ.get("GLAUCOMA_EPOCHS",      "50"))
N_FOLDS       = int(os.environ.get("GLAUCOMA_FOLDS",       "5"))
N_OPTUNA      = int(os.environ.get("GLAUCOMA_TRIALS",      "10"))
N_TUNE_EPOCHS = int(os.environ.get("GLAUCOMA_TUNE_EPOCHS", "3"))
PATIENCE      = int(os.environ.get("GLAUCOMA_PATIENCE",    "8"))
HOLDOUT_FRAC  = float(os.environ.get("GLAUCOMA_HOLDOUT",   "0.20"))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_EXTS      = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# 3.  Data utilities
# ------------------------------------------------------------------
def list_pairs(img_dir: str, mask_dir: str) -> Tuple[List[str], List[str]]:
    """Pair images and masks by file *stem*, robust to ordering."""
    imgs = {os.path.splitext(f)[0]: os.path.join(img_dir, f)
            for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)}
    masks = {os.path.splitext(f)[0]: os.path.join(mask_dir, f)
             for f in os.listdir(mask_dir) if f.lower().endswith(IMG_EXTS)}
    common = sorted(set(imgs) & set(masks))
    if len(common) != len(imgs) or len(common) != len(masks):
        print(f"[warn] {img_dir}: {len(imgs)} imgs, {len(masks)} masks, "
              f"{len(common)} valid pairs")
    return [imgs[k] for k in common], [masks[k] for k in common]


def discretize_mask(mask: np.ndarray) -> np.ndarray:
    """Map a mask (any scale) into class labels {0, 1, 2}."""
    m = mask.astype(np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    out = np.zeros_like(m, dtype=np.uint8)
    out[m < 0.25] = 0
    out[(m >= 0.25) & (m < 0.75)] = 2
    out[m >= 0.75] = 1
    return out


# ------------------------------------------------------------------
# 4.  Augmentation pipeline (identical for all architectures)
# ------------------------------------------------------------------
train_transform = A.Compose([
    A.Resize(*IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5, border_mode=0),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(*IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
    ToTensorV2(),
])
test_transform = val_transform


class GlaucomaDataset(Dataset):
    def __init__(self, images_list, masks_list, transform=None):
        assert len(images_list) == len(masks_list)
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

        mask = imread(self.masks_list[idx], as_gray=True)
        mask = discretize_mask(mask)

        if self.transform:
            t = self.transform(image=img, mask=mask)
            img, mask = t["image"], t["mask"]
        return img.float(), mask.long()


def make_loader(images, masks, transform, batch_size, shuffle):
    ds = GlaucomaDataset(images, masks, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=(device == "cuda"),
                      generator=_G)


# ------------------------------------------------------------------
# 5.  Loss & class weights
# ------------------------------------------------------------------
def compute_class_weights_from_masks(mask_paths: List[str]) -> torch.Tensor:
    """Inverse-frequency weights — computed ONLY from the given fold."""
    flat = []
    for p in mask_paths:
        m = imread(p, as_gray=True)
        flat.append(discretize_mask(m).flatten())
    y = np.concatenate(flat)
    w = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y)
    return torch.tensor(w, dtype=torch.float32, device=device)


def make_loss(dice_w: float, class_weights: torch.Tensor) -> Callable:
    """alpha * Dice + (1 - alpha) * WeightedCE."""
    dice = smp.losses.DiceLoss(mode="multiclass")
    ce   = nn.CrossEntropyLoss(weight=class_weights)
    return lambda p, t: dice_w * dice(p, t) + (1 - dice_w) * ce(p, t)


# ------------------------------------------------------------------
# 6.  Train / validate steps
# ------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn) -> float:
    model.train(); tot = 0.0; n = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(imgs), masks)
        loss.backward(); optimizer.step()
        tot += loss.item() * imgs.size(0); n += imgs.size(0)
    return tot / n


@torch.no_grad()
def validate(model, loader, loss_fn) -> float:
    model.eval(); tot = 0.0; n = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        loss = loss_fn(model(imgs), masks)
        tot += loss.item() * imgs.size(0); n += imgs.size(0)
    return tot / n


# ------------------------------------------------------------------
# 7.  Optuna search (3 epochs, identical search space for all models)
# ------------------------------------------------------------------
def run_optuna(build_model_fn,
               trainval_imgs, trainval_masks,
               optuna_val_imgs, optuna_val_masks,
               n_trials: int = N_OPTUNA) -> dict:

    def objective(trial):
        lr    = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch = trial.suggest_categorical("batch_size", [4, 8])
        dw    = trial.suggest_float("dice_weight", 0.3, 0.7)

        cw = compute_class_weights_from_masks(trainval_masks)
        tr_ld = make_loader(trainval_imgs,  trainval_masks,  train_transform, batch, True)
        vl_ld = make_loader(optuna_val_imgs, optuna_val_masks, val_transform,  batch, False)

        model   = build_model_fn()
        loss_fn = make_loss(dw, cw)
        opt     = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(N_TUNE_EPOCHS):
            train_one_epoch(model, tr_ld, opt, loss_fn)
        v = validate(model, vl_ld, loss_fn)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return v

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print("Best hyper-parameters:", study.best_params)
    return study.best_params


# ------------------------------------------------------------------
# 8.  Cross-validation loop (random KFold, identical for all models)
# ------------------------------------------------------------------
def run_kfold(build_model_fn,
              cv_imgs, cv_masks,
              best_hp: dict, model_dir: str,
              ckpt_prefix: str) -> List[str]:

    os.makedirs(model_dir, exist_ok=True)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_paths = []

    for fold, (tr, vl) in enumerate(kf.split(cv_imgs)):
        print(f"\n========== Fold {fold + 1}/{N_FOLDS} ==========")
        tr_imgs  = [cv_imgs[i]  for i in tr]
        tr_masks = [cv_masks[i] for i in tr]
        vl_imgs  = [cv_imgs[i]  for i in vl]
        vl_masks = [cv_masks[i] for i in vl]

        cw = compute_class_weights_from_masks(tr_masks)
        tr_ld = make_loader(tr_imgs, tr_masks, train_transform,
                            best_hp["batch_size"], True)
        vl_ld = make_loader(vl_imgs, vl_masks, val_transform,
                            best_hp["batch_size"], False)

        model   = build_model_fn()
        loss_fn = make_loss(best_hp["dice_weight"], cw)
        opt     = torch.optim.Adam(model.parameters(), lr=best_hp["lr"])
        sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                       opt, mode="min", factor=0.5, patience=3)

        save_path = os.path.join(model_dir, f"{ckpt_prefix}_fold{fold}.pth")
        best_val  = float("inf"); bad = 0

        for epoch in range(EPOCHS):
            tr_loss = train_one_epoch(model, tr_ld, opt, loss_fn)
            vl_loss = validate(model, vl_ld, loss_fn)
            sched.step(vl_loss)
            print(f"  ep {epoch + 1:02d} | train {tr_loss:.4f} | val {vl_loss:.4f}")

            if vl_loss < best_val - 1e-4:
                best_val, bad = vl_loss, 0
                torch.save(model.state_dict(), save_path)
            else:
                bad += 1
                if bad >= PATIENCE:
                    print(f"  Early stop @ epoch {epoch + 1} (best val={best_val:.4f})")
                    break

        fold_paths.append(save_path)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\nSaved checkpoints:", fold_paths)
    return fold_paths


# ------------------------------------------------------------------
# 9.  Ensemble inference & test-array collection
# ------------------------------------------------------------------
@torch.no_grad()
def ensemble_predict(build_model_fn, model_paths: List[str], loader) -> np.ndarray:
    sum_probs = None
    for path in model_paths:
        model = build_model_fn()
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        probs_list = []
        for imgs, _ in loader:
            imgs = imgs.to(device)
            probs_list.append(torch.softmax(model(imgs), dim=1).cpu().numpy())
        probs_all = np.concatenate(probs_list, axis=0)
        sum_probs = probs_all if sum_probs is None else sum_probs + probs_all
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.argmax(sum_probs / len(model_paths), axis=1)


def collect_test_arrays(loader):
    """Return de-normalised images (CHW, [0,1]) and ground-truth masks."""
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD).reshape(3, 1, 1)
    imgs_all, masks_all = [], []
    for imgs, masks in loader:
        x = imgs.numpy()[0]
        imgs_all.append(x * std + mean)
        masks_all.append(masks.numpy()[0])
    return np.array(imgs_all), np.array(masks_all)


# ------------------------------------------------------------------
# 10.  Metrics
# ------------------------------------------------------------------
def calculate_iou_dice(true_mask, pred_mask, num_classes=NUM_CLASSES):
    iou, dice = [], []
    for c in range(num_classes):
        t = (true_mask == c); p = (pred_mask == c)
        inter = np.logical_and(t, p).sum()
        union = np.logical_or(t, p).sum()
        s = t.sum() + p.sum()
        iou.append(inter / union if union else np.nan)
        dice.append((2 * inter) / s if s else np.nan)
    return np.array(iou), np.array(dice)


def calculate_cdr(mask, cup_class=CUP_CLASS, rim_class=RIM_CLASS):
    cup_area  = np.sum(mask == cup_class)
    disc_area = np.sum((mask == cup_class) | (mask == rim_class))
    return float(cup_area / disc_area) if disc_area else 0.0


def class_metrics(true_mask, pred_mask, num_classes=NUM_CLASSES):
    out = {}
    for c in range(num_classes):
        t = (true_mask == c); p = (pred_mask == c)
        TP = np.logical_and(t, p).sum()
        TN = np.logical_and(~t, ~p).sum()
        FP = np.logical_and(~t, p).sum()
        FN = np.logical_and(t, ~p).sum()
        eps = 1e-8
        acc = (TP + TN) / (TP + TN + FP + FN + eps)
        pre = TP / (TP + FP + eps); rec = TP / (TP + FN + eps)
        f1  = 2 * pre * rec / (pre + rec + eps)
        out[c] = {"Accuracy": acc, "Precision": pre, "Recall": rec, "F1": f1}
    return out


def hausdorff_distance(mask_true, mask_pred, cls):
    a = np.argwhere(mask_true == cls)
    b = np.argwhere(mask_pred == cls)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])


def boundary_iou(mask_true, mask_pred, cls, dilation_size=1):
    tb = binary_dilation(mask_true == cls, disk(dilation_size)) & ~(mask_true == cls)
    pb = binary_dilation(mask_pred == cls, disk(dilation_size)) & ~(mask_pred == cls)
    inter = np.logical_and(tb, pb).sum(); union = np.logical_or(tb, pb).sum()
    return inter / union if union else 0.0


def report_metrics(masks_test: np.ndarray, preds_test: np.ndarray) -> None:
    """Print all metric tables expected by the paper."""
    # ---- IoU / Dice ----
    ious, dices = [], []
    for t, p in zip(masks_test, preds_test):
        i, d = calculate_iou_dice(t, p)
        ious.append(i); dices.append(d)
    ious, dices = np.array(ious), np.array(dices)
    print("\n--- IoU / Dice ---")
    for c in range(NUM_CLASSES):
        print(f"  Class {c}: IoU={np.nanmean(ious[:, c]):.4f}  "
              f"Dice={np.nanmean(dices[:, c]):.4f}")

    # ---- Acc / Prec / Rec / F1 ----
    agg = {c: {k: [] for k in ["Accuracy", "Precision", "Recall", "F1"]}
           for c in range(NUM_CLASSES)}
    for t, p in zip(masks_test, preds_test):
        m = class_metrics(t, p)
        for c in range(NUM_CLASSES):
            for k in agg[c]:
                agg[c][k].append(m[c][k])
    rows = []
    for c in range(NUM_CLASSES):
        row = {"Class": c}
        for k, v in agg[c].items():
            v = np.array(v)
            row[f"{k}_mean"] = v.mean(); row[f"{k}_std"] = v.std()
        rows.append(row)
    print("\n--- Per-class metrics ---")
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- Hausdorff / Boundary IoU ----
    hd_all, bi_all = [], []
    for t, p in zip(masks_test, preds_test):
        hd_all.append([hausdorff_distance(t, p, c) for c in range(NUM_CLASSES)])
        bi_all.append([boundary_iou(t, p, c)       for c in range(NUM_CLASSES)])
    hd_all, bi_all = np.array(hd_all), np.array(bi_all)
    print("\n--- Boundary metrics ---")
    print(pd.DataFrame([{
        "Class": c,
        "Hausdorff_mean": np.nanmean(hd_all[:, c]),
        "Hausdorff_std":  np.nanstd(hd_all[:, c]),
        "BoundaryIoU_mean": np.nanmean(bi_all[:, c]),
        "BoundaryIoU_std":  np.nanstd(bi_all[:, c]),
    } for c in range(NUM_CLASSES)]).to_string(index=False))

    # ---- CDR ----
    cdr_true = np.array([calculate_cdr(m) for m in masks_test])
    cdr_pred = np.array([calculate_cdr(m) for m in preds_test])
    print(f"\nCDR MAE: {np.mean(np.abs(cdr_true - cdr_pred)):.4f}")


# ------------------------------------------------------------------
# 11.  Visualisation (best / worst cases)
# ------------------------------------------------------------------
def plot_case_full(imgs_test, masks_test, preds_test, idx, title=""):
    img       = np.clip(imgs_test[idx].transpose(1, 2, 0), 0, 1)
    true_mask = masks_test[idx]; pred_mask = preds_test[idx]

    iou, dice = calculate_iou_dice(true_mask, pred_mask)
    cdr_t = calculate_cdr(true_mask); cdr_p = calculate_cdr(pred_mask)
    hd = [hausdorff_distance(true_mask, pred_mask, c) for c in range(NUM_CLASSES)]
    bi = [boundary_iou(true_mask, pred_mask, c)       for c in range(NUM_CLASSES)]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img);                   axes[0].set_title("Image"); axes[0].axis("off")
    axes[1].imshow(true_mask, cmap="jet"); axes[1].set_title("GT");    axes[1].axis("off")
    axes[2].imshow(pred_mask, cmap="jet"); axes[2].set_title("Pred");  axes[2].axis("off")
    axes[3].imshow(img); axes[3].set_title("Contours (green=GT, blue=pred)")
    for c in range(1, NUM_CLASSES):
        for ct in measure.find_contours(true_mask == c, 0.5):
            axes[3].plot(ct[:, 1], ct[:, 0], "g", lw=1)
        for ct in measure.find_contours(pred_mask == c, 0.5):
            axes[3].plot(ct[:, 1], ct[:, 0], "b", lw=1)
    axes[3].axis("off")

    txt = f"CDR GT={cdr_t:.3f} | CDR pred={cdr_p:.3f}\n"
    for c in range(NUM_CLASSES):
        txt += (f"Cls{c}: IoU={iou[c]:.3f} Dice={dice[c]:.3f} "
                f"HD={hd[c]:.1f} BIoU={bi[c]:.3f}\n")
    plt.suptitle(f"{title}\n{txt}", fontsize=10)
    plt.tight_layout(); plt.show()


def visualise_best_worst(imgs_test, masks_test, preds_test, k: int = 5):
    """Plot the top-k best and worst cases (ranked by mean Dice of cup+disc)."""
    dices = np.array([calculate_iou_dice(t, p)[1] for t, p
                       in zip(masks_test, preds_test)])
    scores    = np.nanmean(dices[:, 1:], axis=1)
    order     = np.argsort(scores)
    worst_idx = order[:k]
    best_idx  = order[-k:][::-1]
    print(f"\n===== TOP-{k} BEST =====")
    for i in best_idx:
        plot_case_full(imgs_test, masks_test, preds_test, i, f"Best (idx={i})")
    print(f"\n===== TOP-{k} WORST =====")
    for i in worst_idx:
        plot_case_full(imgs_test, masks_test, preds_test, i, f"Worst (idx={i})")


# ------------------------------------------------------------------
# 12.  End-to-end runner used by every architecture script
# ------------------------------------------------------------------
def run_full_pipeline(build_model_fn: Callable, model_dir: str,
                      ckpt_prefix: str,
                      train_img_dir: str, train_mask_dir: str,
                      test_img_dir: str,  test_mask_dir: str) -> None:
    """Run Optuna -> 5-fold CV -> test ensemble -> metrics -> figures."""
    print(f"Device: {device}")
    train_imgs, train_masks = list_pairs(train_img_dir, train_mask_dir)
    test_imgs,  test_masks  = list_pairs(test_img_dir,  test_mask_dir)
    print(f"Train: {len(train_imgs)} | Test: {len(test_imgs)}")

    # Holdout for Optuna (fixed, never touched by CV)
    trainval_idx, holdout_idx = train_test_split(
        np.arange(len(train_imgs)),
        test_size=HOLDOUT_FRAC, random_state=SEED, shuffle=True,
    )
    cv_imgs    = [train_imgs[i]  for i in trainval_idx]
    cv_masks   = [train_masks[i] for i in trainval_idx]
    opt_imgs   = [train_imgs[i]  for i in holdout_idx]
    opt_masks  = [train_masks[i] for i in holdout_idx]
    print(f"CV pool: {len(cv_imgs)} | Optuna holdout: {len(opt_imgs)}")

    # 1) Optuna search
    best_hp = run_optuna(build_model_fn, cv_imgs, cv_masks, opt_imgs, opt_masks)

    # 2) 5-fold CV
    fold_paths = run_kfold(build_model_fn, cv_imgs, cv_masks,
                           best_hp, model_dir, ckpt_prefix)

    # 3) Test ensemble
    test_ds = GlaucomaDataset(test_imgs, test_masks, test_transform)
    test_ld = DataLoader(test_ds, batch_size=1, shuffle=False,
                         num_workers=2, pin_memory=(device == "cuda"))
    imgs_test, masks_test = collect_test_arrays(test_ld)
    preds_test = ensemble_predict(build_model_fn, fold_paths, test_ld)
    print("Ensemble predictions:", preds_test.shape)

    # 4) Metrics & visualisation
    report_metrics(masks_test, preds_test)
    visualise_best_worst(imgs_test, masks_test, preds_test, k=5)
