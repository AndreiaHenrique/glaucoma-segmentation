"""
UPerNet (encoder = ResNet-34, ImageNet pre-trained).

The full training / evaluation pipeline is shared with `manet_optuna.py`
and `dpt_optuna.py` through `pipeline.py`. The ONLY architecture-
specific code is `build_model()` below, which guarantees a fair
comparison between the three architectures described in the paper.

Paths are read from environment variables so the script is portable:

    GLAUCOMA_TRAIN_IMG   default: ./data/Segmentacao/train/image
    GLAUCOMA_TRAIN_MASK  default: ./data/Segmentacao/train/mask
    GLAUCOMA_TEST_IMG    default: ./data/Segmentacao/test/image
    GLAUCOMA_TEST_MASK   default: ./data/Segmentacao/test/mask
    UPERNET_MODEL_DIR    default: ./checkpoints/upernet_models
"""
import os
import segmentation_models_pytorch as smp

from pipeline import NUM_CLASSES, device, run_full_pipeline


def build_model():
    """UPerNet with ResNet-34 encoder (ImageNet pre-trained)."""
    return smp.UPerNet(
        encoder_name    = "resnet34",
        encoder_depth   = 5,
        encoder_weights = "imagenet",
        decoder_channels= 256,
        decoder_use_norm= "batchnorm",
        in_channels     = 3,
        classes         = NUM_CLASSES,
        activation      = None,
        upsampling      = 4,
    ).to(device)


if __name__ == "__main__":
    run_full_pipeline(
        build_model_fn = build_model,
        model_dir      = os.environ.get("UPERNET_MODEL_DIR",
                                        "./checkpoints/upernet_models"),
        ckpt_prefix    = "upernet",
        train_img_dir  = os.environ.get("GLAUCOMA_TRAIN_IMG",
                                        "./data/Segmentacao/train/image"),
        train_mask_dir = os.environ.get("GLAUCOMA_TRAIN_MASK",
                                        "./data/Segmentacao/train/mask"),
        test_img_dir   = os.environ.get("GLAUCOMA_TEST_IMG",
                                        "./data/Segmentacao/test/image"),
        test_mask_dir  = os.environ.get("GLAUCOMA_TEST_MASK",
                                        "./data/Segmentacao/test/mask"),
    )
