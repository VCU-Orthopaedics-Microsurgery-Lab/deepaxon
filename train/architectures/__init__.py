from train.architectures.unet_plus_plus import build_unet_plus_plus
from train.architectures.unet import build_unet


def build_model(
    model_type: str = 'unet++',
    input_shape: tuple = (256, 256, 1),
    n_classes: int = 3,
    filters: int = 16
):
    """
    Factory function for DeepAxon segmentation architectures.
    model_type: 'unet++' (default) or 'unet'
    """
    if model_type == 'unet++':
        return build_unet_plus_plus(input_shape, n_classes, filters)
    elif model_type == 'unet':
        return build_unet(input_shape, n_classes, filters)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Use 'unet++' or 'unet'.")