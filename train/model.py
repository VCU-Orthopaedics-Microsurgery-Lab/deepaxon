'''
-------------------------------- DEEPAXON --------------------------------
model.py — Contains the convolutional neural networks for training DeepAxon segmentation models.

This script provides:
- Base U-Net architecture
- UNet++ (DeepAxon++) architecture
- Utility blocks for encoder and decoder
- Factory function to select model type
'''

# [M] Added BatchNormalization and L2 regularization support
# to improve training stability and reduce overfitting.


# ------------------------------ Keras Imports ------------------------------------------ #
# Import necessary modules for Keras-based Deep Learning architecture development
from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose,
    Dropout, BatchNormalization
)
from keras.regularizers import l2 # L2 regularization to reduce overfitting

# ------------------------------ Convolutional Blocks ----------------------------------- #
# [M] base_filters parameter introduced for flexible scaling of model capacity
# (e.g., doubling filters globally for higher model complexity)
def conv_block(inputs, filters, dropout=0.1, kernel_reg=1e-4):
    '''
    2-stage convolutional block with BatchNormalization and Dropout.
    Used in encoder/downscaling paths.
    
    :param inputs: Tensor; input
    :param filters: Integer; the dimensionality of the output space (i.e. the number of output filters in the convolution).
    :param dropout: Integer; dropout value for the dropout between convolutional layers (default = 0.1)
    :param kernel_reg: float; L2 regularization coefficient (default = 1e-4)
    
    :returns: Tensor; a tensor of rank 4+
    
    #[M] Added BatchNormalization after each Conv2D
    #[M] Added L2 regularization with default 1e-4
    '''
    
    c1 = Conv2D(filters, (3, 3), activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(kernel_reg),
                padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(dropout)(c1)

    c2 = Conv2D(filters, (3, 3), activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(kernel_reg),
                padding='same')(c1)
    c2 = BatchNormalization()(c2)

    return c2

def exp_block(up, skip, filters, dropout=0.1, kernel_reg=1e-4):
    '''
    Expansive block: upscaling + concatenation + 2 convolutional layers.
    Used in decoder/upscaling paths.
    
    :param up: Tensor; a single tensor that is below the current node of interest. Will be used in the upscaling process.
    :param skip: List of Tensor objects; all tensors that are on the same level as the current node of interest. Will be used in the concatenation process.
    :param filters: Integer; the dimensionality of the output space (i.e. the number of output filters in the convolution).
    :param dropout: Integer; dropout value for the dropout between convolutional layers; Default = 0.1
    :param kernel_reg: float; L2 regularization coefficient (default = 1e-4)
    
    :returns: Tensor; a tensor of rank 4+
    
    #[M] skip + [e1] ensures a new list is created (prevents in-place modification of skip list)
    #[M] Added BatchNormalization and L2 regularization as in conv_block
    '''
    # Upsample
    e1 = Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(up)
    g2 = concatenate(skip + [e1]) #[M] Concatenate skip connections safely without modifying the original skip lis

    c1 = Conv2D(filters, (3, 3), activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(kernel_reg),
                padding='same')(g2)
    c1 = BatchNormalization()(c1)   #[M] Added Normalization
    c1 = Dropout(dropout)(c1)

    c2 = Conv2D(filters, (3, 3), activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(kernel_reg),
                padding='same')(c1)
    c2 = BatchNormalization()(c2)   #[M] Added Normalization

    return c2

# ------------------------------ UNet++ (DeepAxon++) Model--------------------------------------------- #
def deepaxon_plusplus_model(input_shape=(256, 256, 1), num_classes=3, base_filters=16):
    '''
    UNet++ architecture for multi-class segmentation.
    
    Args:
    :param input_shape (tuple): Input shape (H, W, C); Default = (256, 256, 1)
    :param num_classes (int): Number of segmentation classes [For segmented nerve images, the number of classes is 3 (background, myelin, axons)]
    :param base_filters (int): Number of filters in the first layer
    
    :returns: Keras Model Object; A model grouping layers into an object with training/inference features.
    Once the model is created, you can config the model with losses and metrics with model.compile(), train the model with model.fit(),
    or use the model to do prediction with model.predict().
    '''
    inputs = Input(input_shape)
    
    # Encoder
    x00 = conv_block(inputs, base_filters, dropout=0.1)
    x10 = conv_block(MaxPooling2D((2, 2))(x00), base_filters * 2, dropout=0.1)
    x20 = conv_block(MaxPooling2D((2, 2))(x10), base_filters * 4, dropout=0.2)
    x30 = conv_block(MaxPooling2D((2, 2))(x20), base_filters * 8, dropout=0.2)
    x40 = conv_block(MaxPooling2D((2, 2))(x30), base_filters * 16, dropout=0.3)
    
    # Decoder (nested skip connections)
    x01 = exp_block(up=x10, skip=[x00], filters=base_filters, dropout=0.1)
    x11 = exp_block(up=x20, skip=[x10], filters=base_filters * 2, dropout=0.1)
    x21 = exp_block(up=x30, skip=[x20], filters=base_filters * 4, dropout=0.2)
    x31 = exp_block(up=x40, skip=[x30], filters=base_filters * 8, dropout=0.2)
    
    x02 = exp_block(up=x11, skip=[x00, x01], filters=base_filters, dropout=0.1)
    x12 = exp_block(up=x21, skip=[x10, x11], filters=base_filters * 2, dropout=0.1)
    x22 = exp_block(up=x31, skip=[x20, x21], filters=base_filters * 4, dropout=0.2)
    
    x03 = exp_block(up=x12, skip=[x00, x01, x02], filters=base_filters, dropout=0.1)
    x13 = exp_block(up=x22, skip=[x10, x11, x12], filters=base_filters * 2, dropout=0.1)
    
    x04 = exp_block(up=x13, skip=[x00, x01, x02, x03], filters=base_filters, dropout=0.1)
    
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    outputs = Conv2D(num_classes, (1, 1), activation=activation)(x04)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# ------------------------------ Base U-Net model ---------------------------------------- #
def deepaxon_model(input_shape=(256, 256, 1), num_classes=3, base_filters=16):
    '''
    Base U-Net architecture
    
    :param input_shape: Tuple; format is (IMAGE HEIGHT, IMAGE WIDTH, IMAGE CHANNELS); Default = (256, 256, 1)
    :param num_classes: Integer; number of classes for multi-class segmentation.
    For segmented nerve images, the number of classes is 3 (background, myelin, axons)
    
    :returns: Keras Model Object; A model grouping layers into an object with training/inference features.
    Once the model is created, you can config the model with losses and metrics with model.compile(), train the model with model.fit(),
    or use the model to do prediction with model.predict().
    '''
    inputs = Input(input_shape)
    
    # Encoder
    x00 = conv_block(inputs, 16, dropout=0.1)
    x10 = conv_block(MaxPooling2D((2, 2))(x00), base_filters * 2, dropout=0.1)
    x20 = conv_block(MaxPooling2D((2, 2))(x10), base_filters * 4, dropout=0.2)
    x30 = conv_block(MaxPooling2D((2, 2))(x20), base_filters * 8, dropout=0.2)
    x40 = conv_block(MaxPooling2D((2, 2))(x30), base_filters * 16, dropout=0.3)
    
    # Decoder
    x31 = exp_block(up=x40, skip=[x30], filters=base_filters * 8, dropout=0.2)
    x21 = exp_block(up=x31, skip=[x20], filters=base_filters * 4, dropout=0.2)
    x11 = exp_block(up=x21, skip=[x10], filters=base_filters * 2, dropout=0.1)
    x01 = exp_block(up=x11, skip=[x00], filters=base_filters, dropout=0.1)
    
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    outputs = Conv2D(num_classes, (1, 1), activation=activation)(x01)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# ----------------------------- Model Selector Function --------------------------- #
#[M] Added build_model() factory function
# to unify deepaxon and deepaxon++ architectures, and to automatically
# select the final activation function (sigmoid for binary, softmax for multi-class)

def build_model(model_type='unet++', input_shape=(256,256,1), num_classes=3, base_filters=16):
    '''
    [M] Factory function to build either DeepAxon or DeepAxon++
    [M] Auto-chooses activation function based on num_classes
    [M] Flexible base_filters allows easy scaling of architecture
    '''
    if model_type == 'unet++':
        model = deepaxon_plusplus_model(input_shape, num_classes, base_filters)
    else:
        model = deepaxon_model(input_shape, num_classes, base_filters)
    
    return model
    
# ----------------------------- Quick test ------------------------------ #
#[M] Added model summary printout for verification
# when running the script directly (e.g., python model.py)

if __name__ == "__main__":
    model = build_model(architecture="unet++", input_shape=(256, 256, 1), num_classes=3)
    model.summary()