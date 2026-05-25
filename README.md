# Comparative Evaluation of Deep Learning Architectures for Glaucoma Screening

This repository accompanies the paper **"Comparative Evaluation of Deep Learning Architectures for Glaucoma Screening: From Internal Accuracy to Cross-Domain"** by *Andreia Henrique*.

Three semantic-segmentation architectures are compared on optic cup / optic disc segmentation in colour fundus images:

| Architecture | Encoder backbone | Library |
|---|---|---|
| **UPerNet** | ResNet-34 (ImageNet) | `segmentation-models-pytorch` |
| **MAnet** | ResNet-34 (ImageNet) | `segmentation-models-pytorch` |
| **DPT** (Dense Prediction Transformer) | ViT-Base/16 (`vit_base_patch16_224`) | `segmentation-models-pytorch` + `timm` |

The best model from the internal evaluation is then validated zero-shot on the **REFUGE** dataset.

---

## 1. Fair comparison by design

All three architectures share **exactly the same training and evaluation pipeline** through `src/pipeline.py`.
The architecture wrappers (`upernet_optuna.py`, `manet_optuna.py`, `dpt_optuna.py`) only define a `build_model()` function — everything else (data, augmentation, loss, Optuna search space, K-fold strategy, optimiser, scheduler, early stopping, ensemble, metrics) is identical.

| Stage | Setting |
|---|---|
| Input resolution | 224 × 224 |
| Classes | 0 = background · 1 = cup · 2 = disc / rim |
| Augmentation | HFlip(0.5) · VFlip(0.5) · Rotate(±15°, p=0.5) · BrightnessContrast(p=0.2) · ImageNet normalisation |
| Loss | `L = α · Dice + (1 − α) · WeightedCrossEntropy` (class weights = inverse frequency, per fold) |
| Optuna (TPE, seed = 42, 10 trials, 3 epochs) | `lr ∈ [1e-5, 1e-3]` log · `batch ∈ {4, 8}` · `α ∈ [0.3, 0.7]` |
| Optuna holdout | 20 % of the training pool (fixed, never touched by CV) |
| Cross-validation | `KFold(5)`, random, seed = 42 |
| Optimiser | Adam |
| Scheduler | `ReduceLROnPlateau(factor=0.5, patience=3)` |
| Early stopping | patience = 8 epochs on val loss |
| Epochs | 50 per fold |
| Ensemble | mean of softmax across the 5 folds → `argmax` |

---

## 2. Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── pipeline.py          # shared training / evaluation pipeline
│   ├── upernet_optuna.py    # UPerNet wrapper (only `build_model`)
│   ├── manet_optuna.py      # MAnet   wrapper
│   ├── dpt_optuna.py        # DPT     wrapper
│   └── test_refuge.py       # zero-shot evaluation on REFUGE (Oracle crop)
|   └── predict_masks.py     # making annotation for datasets
└── docs/
    ├── VSCODE_RUN_GUIDE.md      # steps to run in vscode
```

---

## 3. Dataset layout expected by the scripts

```
data/
└── Segmentacao/
    ├── train/
    │   ├── image/    G-01-L.png ...
    │   └── mask/     G-01-L.png ...
    └── test/
        ├── image/
        └── mask/
```

For the REFUGE external set:

```
data/
└── REFUGE/
    └── test/
        ├── images/
        └── mask/
```

Masks are single-channel images encoding the three regions through different intensities;
`pipeline.discretize_mask()` maps them to `{0, 1, 2}`. The REFUGE script uses its own
discretisation that accounts for the REFUGE encoding (white = background, grey = disc,
black = cup).

> Hard-coded Google Drive paths from the original Colab notebooks have been replaced by
> environment variables. The defaults match the layout above, so no change is needed if
> you keep the suggested folder structure.

---

## 4. Reproduce the experiments

```bash
# 1) Create environment (Python 3.10+ recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train each architecture (≈6 h per arch on an RTX 3060)
python src/upernet_optuna.py
python src/manet_optuna.py
python src/dpt_optuna.py

# 4) External validation on REFUGE (uses UPerNet checkpoints)
python src/test_refuge.py
```

Override paths with environment variables, e.g.

```bash
# Linux / macOS
export GLAUCOMA_TRAIN_IMG=/path/to/train/image
# Windows PowerShell
$env:GLAUCOMA_TRAIN_IMG = "C:\path\to\train\image"
```

A GPU with **≥ 8 GB VRAM** is recommended.

> **Note.** The numbers reported in the paper were obtained with the original Colab
> notebooks. After the refactor, the pipeline was harmonised across the three
> architectures to remove minor inconsistencies.
> Re-running the scripts on the same data should reproduce the results up to small
> stochastic variation typical of deep learning experiments.

---

## 5. Main results

### Internal test set

| Model    | Cup IoU | Cup Dice | Disc IoU | Disc Dice |
|----------|--------:|---------:|---------:|----------:|
| MAnet    | 0.7805  | 0.8690   | 0.8091   | 0.8921    |
| **UPerNet** | **0.7893** | **0.8750** | **0.8119** | **0.8941** |
| DPT      | 0.7853  | 0.8729   | 0.8027   | 0.8883    |

| Model    | CDR MAE |
|----------|--------:|
| MAnet    | 0.1007  |
| **UPerNet** | **0.0849** |
| DPT      | 0.0954  |

### Zero-shot on REFUGE (UPerNet + Oracle crop)

| Structure | IoU    | Dice   | HD   | BIoU   |
|-----------|-------:|-------:|-----:|-------:|
| Cup       | 0.7879 | 0.8791 | 5.90 | 0.0501 |
| Disc      | 0.7788 | 0.8737 | 7.10 | 0.0409 |

CDR MAE on REFUGE: **0.0752**.

---

## 6. Generating masks for a new (unlabeled) dataset
# Linux/macOS
NEW_IMG_DIR=./data/my_dataset python src/predict_masks.py
# Windows PowerShell
$env:NEW_IMG_DIR="./data/my_dataset"; python src/predict_masks.py
## 8. License

Released under the [MIT License](LICENSE).
