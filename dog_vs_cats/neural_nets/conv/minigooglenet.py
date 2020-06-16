# Coded by u301671
# import all the necessary packages

from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Softmax, Input
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K


class MiniGoogLeNet:
    @staticmethod
    def conv_module(x, k, kX, kY, stride, chanDim, padding="same"):
        # CONV -- BN --- ACT(RELU)
        x = Conv2D(k, (kX, kY), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        # Return the block
        return x

    @staticmethod
    def Inception_Module(x, numK1x1, numK3x3, chanDim):
        # CONV1 --- CONV2 ---- Concatenate
        conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
        conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
        x = concatenate([conv_1x1, conv_3x3], axis=chanDim)
        # Return the block
        return x

    @staticmethod
    def Downsample_Module(x, k, chanDim):
        # CONV --- Pool ---- Concatenate (Parallel Processing)
        conv_3x3 = MiniGoogLeNet.conv_module(x, k, 3, 3, (2, 2), chanDim,
                                             padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chanDim)
        # Return the block
        return x

    # Pull all the pieces together
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        inputs = Input(shape=inputShape)
        print("Input Shape: ", inputShape)
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)
        x = MiniGoogLeNet.Inception_Module(x, 32, 32, chanDim)
        x = MiniGoogLeNet.Inception_Module(x, 32, 48, chanDim)
        x = MiniGoogLeNet.Downsample_Module(x, 80, chanDim)

        x = MiniGoogLeNet.Inception_Module(x, 112, 48, chanDim)
        x = MiniGoogLeNet.Inception_Module(x, 96, 64, chanDim)
        x = MiniGoogLeNet.Inception_Module(x, 80, 80, chanDim)
        x = MiniGoogLeNet.Inception_Module(x, 48, 96, chanDim)
        x = MiniGoogLeNet.Downsample_Module(x, 96, chanDim)

        x = MiniGoogLeNet.Inception_Module(x, 176, 160, chanDim)
        x = MiniGoogLeNet.Inception_Module(x, 176, 160, chanDim)

        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        # Softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # Create the model
        model = Model(inputs, x, name="googlenet")
        # Return the model
        return model
