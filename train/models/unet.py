"""
train/models/unet.py

Standard UNet architecture for DeepAxon.
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


def exp_block(x, skip, filters: int):
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def UNET(input_shape=(256, 256, 1), n_classes: int = 3, filters: int = 16) -> Model:
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, filters * 1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, filters * 2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, filters * 4)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, filters * 8)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    c5 = conv_block(p4, filters * 16)

    # Decoder
    u6 = exp_block(c5, c4, filters * 8)
    u7 = exp_block(u6, c3, filters * 4)
    u8 = exp_block(u7, c2, filters * 2)
    u9 = exp_block(u8, c1, filters * 1)

    outputs = layers.Conv2D(n_classes, 1, activation='softmax')(u9)
    return Model(inputs, outputs, name="UNET")
