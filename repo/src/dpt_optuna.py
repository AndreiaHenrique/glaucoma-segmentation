"""
DPT — Dense Prediction Transformer
(encoder = ViT-Base/16, `vit_base_patch16_224.augreg_in21k`).

The full training / evaluation pipeline is shared with `upernet_optuna.py`
and `manet_optuna.py` through `pipeline.py`. The ONLY architecture-
specific code is `build_model()` below, which guarantees a fair
comparison between the three architectures described in the paper.

Paths are read from environment variables (same names as the UPerNet
script). Checkpoints go to `DPT_MODEL_DIR`
(default: ./checkpoints/dpt_models).
"""
import os
import segmentation_models_pytorch as smp

from pipeline import NUM_CLASSES, device, run_full_pipeline


def build_model():
    """DPT with ViT-Base/16 encoder (ImageNet pre-trained)."""
    return smp.DPT(
        encoder_name    = "tu-vit_base_patch16_224.augreg_in21k",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = NUM_CLASSES,
    ).to(device)


if __name__ == "__main__":
    run_full_pipeline(
        build_model_fn = build_model,
        model_dir      = os.environ.get("DPT_MODEL_DIR",
                                        "./checkpoints/dpt_models"),
        ckpt_prefix    = "dpt",
        train_img_dir  = os.environ.get("GLAUCOMA_TRAIN_IMG",
                                        "./data/Segmentacao/train/image"),
        train_mask_dir = os.environ.get("GLAUCOMA_TRAIN_MASK",
                                        "./data/Segmentacao/train/mask"),
        test_img_dir   = os.environ.get("GLAUCOMA_TEST_IMG",
                                        "./data/Segmentacao/test/image"),
        test_mask_dir  = os.environ.get("GLAUCOMA_TEST_MASK",
                                        "./data/Segmentacao/test/mask"),
    )
