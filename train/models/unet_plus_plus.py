# train/models/unet_plus_plus.py
"""
UNET_PLUS_PLUS (DeepAxon++) architecture and factory function.
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D
from .unet import conv_block, exp_block, UNET

# ------------------------------ UNET_PLUS_PLUS --------------------------------------------- #
def UNET_PLUS_PLUS(input_shape=(256,256,1), num_classes=3, base_filters=16):
    inputs = Input(input_shape)
    # Encoder
    x00 = conv_block(inputs, base_filters, dropout=0.1)
    x10 = conv_block(MaxPooling2D((2,2))(x00), base_filters*2, dropout=0.1)
    x20 = conv_block(MaxPooling2D((2,2))(x10), base_filters*4, dropout=0.2)
    x30 = conv_block(MaxPooling2D((2,2))(x20), base_filters*8, dropout=0.2)
    x40 = conv_block(MaxPooling2D((2,2))(x30), base_filters*16, dropout=0.3)

    # Decoder (nested skip connections)
    x01 = exp_block(up=x10, skip=[x00], filters=base_filters, dropout=0.1)
    x11 = exp_block(up=x20, skip=[x10], filters=base_filters*2, dropout=0.1)
    x21 = exp_block(up=x30, skip=[x20], filters=base_filters*4, dropout=0.2)
    x31 = exp_block(up=x40, skip=[x30], filters=base_filters*8, dropout=0.2)

    x02 = exp_block(up=x11, skip=[x00,x01], filters=base_filters, dropout=0.1)
    x12 = exp_block(up=x21, skip=[x10,x11], filters=base_filters*2, dropout=0.1)
    x22 = exp_block(up=x31, skip=[x20,x21], filters=base_filters*4, dropout=0.2)

    x03 = exp_block(up=x12, skip=[x00,x01,x02], filters=base_filters, dropout=0.1)
    x13 = exp_block(up=x22, skip=[x10,x11,x12], filters=base_filters*2, dropout=0.1)

    x04 = exp_block(up=x13, skip=[x00,x01,x02,x03], filters=base_filters, dropout=0.1)

    activation = 'sigmoid' if num_classes==1 else 'softmax'
    outputs = Conv2D(num_classes, (1,1), activation=activation)(x04)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# ----------------------------- Factory Function --------------------------- #
def build_model(model_type='unet++', input_shape=(256,256,1), num_classes=3, base_filters=16):
    if model_type.lower() in ['unet++','deepaxon++']:
        return UNET_PLUS_PLUS(input_shape, num_classes, base_filters)
    else:
        return UNET(input_shape, num_classes, base_filters)