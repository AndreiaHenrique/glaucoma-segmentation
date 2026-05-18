"""
MAnet (encoder = ResNet-34, ImageNet pre-trained).

The full training / evaluation pipeline is shared with `upernet_optuna.py`
and `dpt_optuna.py` through `pipeline.py`. The ONLY architecture-
specific code is `build_model()` below, which guarantees a fair
comparison between the three architectures described in the paper.

Paths are read from environment variables (same names as the UPerNet
script). Checkpoints go to `MANET_MODEL_DIR`
(default: ./checkpoints/manet_models).
"""
import os
import segmentation_models_pytorch as smp

from pipeline import NUM_CLASSES, device, run_full_pipeline


def build_model():
    """MAnet with ResNet-34 encoder (ImageNet pre-trained)."""
    return smp.MAnet(
        encoder_name        = "resnet34",
        encoder_depth       = 5,
        encoder_weights     = "imagenet",
        decoder_use_norm    = "batchnorm",
        decoder_channels    = (256, 128, 64, 32, 16),
        decoder_pab_channels= 64,
        decoder_interpolation = "nearest",
        in_channels         = 3,
        classes             = NUM_CLASSES,
        activation          = None,
        aux_params          = None,
    ).to(device)


if __name__ == "__main__":
    run_full_pipeline(
        build_model_fn = build_model,
        model_dir      = os.environ.get("MANET_MODEL_DIR",
                                        "./checkpoints/manet_models"),
        ckpt_prefix    = "manet",
        train_img_dir  = os.environ.get("GLAUCOMA_TRAIN_IMG",
                                        "./data/Segmentacao/train/image"),
        train_mask_dir = os.environ.get("GLAUCOMA_TRAIN_MASK",
                                        "./data/Segmentacao/train/mask"),
        test_img_dir   = os.environ.get("GLAUCOMA_TEST_IMG",
                                        "./data/Segmentacao/test/image"),
        test_mask_dir  = os.environ.get("GLAUCOMA_TEST_MASK",
                                        "./data/Segmentacao/test/mask"),
    )
