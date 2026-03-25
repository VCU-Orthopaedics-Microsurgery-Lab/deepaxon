"""
train/architectures/unet_plus_plus.py

UNet++ (nested skip connections) — DeepAxon's primary architecture.
"""

from tensorflow.keras import layers, Model


def conv_block(x, filters: int, dropout: float = 0.1):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def upsample_concat(x, *skips, filters: int):
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, *skips])
    x = conv_block(x, filters)
    return x


def build_unet_plus_plus(
    input_shape: tuple = (256, 256, 1),
    n_classes: int = 3,
    filters: int = 16
) -> Model:
    """
    UNet++ with dense nested skip connections.
    Node notation: x{depth}{column} following Zhou et al. 2018.

    filters=16: lightweight base chosen for 256×256 single-channel input.
    Gives 16→32→64→128→256 progression — sufficient capacity for 3-class
    axon/myelin segmentation without overfitting small training sets.
    """
    inputs = layers.Input(shape=input_shape)
    f = filters

    # ── Encoder (column 0) ───────────────────────────────────────────────────
    x00 = conv_block(inputs, f * 1)
    p0  = layers.MaxPooling2D()(x00)

    x10 = conv_block(p0, f * 2)
    p1  = layers.MaxPooling2D()(x10)

    x20 = conv_block(p1, f * 4)
    p2  = layers.MaxPooling2D()(x20)

    x30 = conv_block(p2, f * 8)
    p3  = layers.MaxPooling2D()(x30)

    x40 = conv_block(p3, f * 16)   # bottleneck

    # ── Nested decoder nodes ─────────────────────────────────────────────────
    # Column 1
    x01 = upsample_concat(x10, x00, filters=f * 1)
    x11 = upsample_concat(x20, x10, filters=f * 2)
    x21 = upsample_concat(x30, x20, filters=f * 4)
    x31 = upsample_concat(x40, x30, filters=f * 8)

    # Column 2
    x02 = upsample_concat(x11, x00, x01, filters=f * 1)
    x12 = upsample_concat(x21, x10, x11, filters=f * 2)
    x22 = upsample_concat(x31, x20, x21, filters=f * 4)

    # Column 3
    x03 = upsample_concat(x12, x00, x01, x02, filters=f * 1)
    x13 = upsample_concat(x22, x10, x11, x12, filters=f * 2)

    # Column 4
    x04 = upsample_concat(x13, x00, x01, x02, x03, filters=f * 1)

    outputs = layers.Conv2D(n_classes, 1, activation='softmax')(x04)
    return Model(inputs, outputs, name="UNET_PLUS_PLUS")