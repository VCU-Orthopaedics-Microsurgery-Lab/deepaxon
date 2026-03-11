# train/models/unet.py
"""
UNET model architecture and convolutional blocks for DeepAxon.
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization, concatenate
from keras.regularizers import l2

# ------------------------------ Convolutional Blocks ----------------------------------- #
def conv_block(inputs, filters, dropout=0.1, kernel_reg=1e-4):
    c1 = Conv2D(filters, (3,3), activation='relu', kernel_initializer='he_normal',
                kernel_regularizer=l2(kernel_reg), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(dropout)(c1)

    c2 = Conv2D(filters, (3,3), activation='relu', kernel_initializer='he_normal',
                kernel_regularizer=l2(kernel_reg), padding='same')(c1)
    c2 = BatchNormalization()(c2)
    return c2

def exp_block(up, skip, filters, dropout=0.1, kernel_reg=1e-4):
    e1 = Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(up)
    g2 = concatenate(skip + [e1])

    c1 = Conv2D(filters, (3,3), activation='relu', kernel_initializer='he_normal',
                kernel_regularizer=l2(kernel_reg), padding='same')(g2)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(dropout)(c1)

    c2 = Conv2D(filters, (3,3), activation='relu', kernel_initializer='he_normal',
                kernel_regularizer=l2(kernel_reg), padding='same')(c1)
    c2 = BatchNormalization()(c2)
    return c2

# ------------------------------ Base UNET ---------------------------------------- #
def UNET(input_shape=(256,256,1), num_classes=3, base_filters=16):
    inputs = Input(input_shape)
    x00 = conv_block(inputs, base_filters, dropout=0.1)
    x10 = conv_block(MaxPooling2D((2,2))(x00), base_filters*2, dropout=0.1)
    x20 = conv_block(MaxPooling2D((2,2))(x10), base_filters*4, dropout=0.2)
    x30 = conv_block(MaxPooling2D((2,2))(x20), base_filters*8, dropout=0.2)
    x40 = conv_block(MaxPooling2D((2,2))(x30), base_filters*16, dropout=0.3)

    x31 = exp_block(up=x40, skip=[x30], filters=base_filters*8, dropout=0.2)
    x21 = exp_block(up=x31, skip=[x20], filters=base_filters*4, dropout=0.2)
    x11 = exp_block(up=x21, skip=[x10], filters=base_filters*2, dropout=0.1)
    x01 = exp_block(up=x11, skip=[x00], filters=base_filters, dropout=0.1)

    activation = 'sigmoid' if num_classes==1 else 'softmax'
    outputs = Conv2D(num_classes, (1,1), activation=activation)(x01)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model