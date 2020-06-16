# coded by u301671
# Import the necessary packages

from sklearn.preprocessing import LabelBinarizer
from catsanddogs_kaggle.dog_vs_cats.neural_nets.conv.minigooglenet import MiniGoogLeNet
from catsanddogs_kaggle.dog_vs_cats.callbacks.trainingmonitor import TrainingMonitor
from catsanddogs_kaggle.dog_vs_cats.callbacks.epochcheckpoint import EpochCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import pandas as pd
import argparse
import os
from keras import backend as K
import matplotlib
matplotlib.use("Agg")


# Path to output model
OUTPUT_PATH = "./catsanddogs_kaggle/dog_vs_cats/output"
CHECKPOINT_PATH = "./catsanddogs_kaggle/dog_vs_cats/output/checkpoints/"
MODEL_PATH = "./catsanddogs_kaggle/dog_vs_cats/output/minigooglenet_cifar10.hdf5"

start_Epoch = 0
NUM_EPOCHS = 70
base_alpha = 5e-3


# Define the polynomial learning rate function
def poly_learning_rate(epoch_curr):
    epoch_max = NUM_EPOCHS
    alpha_init = base_alpha
    poly_power = 1.0

    alpha = alpha_init * (1 - (epoch_curr / float(epoch_max))) ** poly_power
    return alpha


# Load the training and testing data. Convert the images from integers to float
print("[INFO] loading training and testing data....")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")
print("TrainX: ", trainX.shape)
print("TrainY: ", trainY.shape)

# Apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# Construct the image generator for data augmentation
imagegen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                              horizontal_flip=True, fill_mode="nearest")

# Construct the set of callbacks
# checkPoints = os.path.sep.join([CHECKPOINT_PATH, "{}.txt".format(os.getpid())])
figPath = os.path.sep.join([OUTPUT_PATH, "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([OUTPUT_PATH, "{}.json".format(os.getpid())])
callbacks = [EpochCheckpoint(CHECKPOINT_PATH, every=1, startAt=start_Epoch),
             TrainingMonitor(figPath, jsonPath=jsonPath),
             LearningRateScheduler(poly_learning_rate)]

# Initialize the optimizer and model
if len(os.listdir(CHECKPOINT_PATH)) == 0:
    print("[INFO] Compiling model..........")
    opt = SGD(lr=base_alpha, momentum=0.9)
    model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
else:
    # Load the checkpoints from the disk
    lst = os.listdir(CHECKPOINT_PATH)
    matching = [s for s in lst if '.hdf5' in s]
    checkpoint_latest = matching[0]
    checkpoint_fullpath = os.path.join(CHECKPOINT_PATH, checkpoint_latest)
    print("Checkpoint fullpath:", checkpoint_fullpath)
    print("[INFO] loading {}...".format(os.listdir(CHECKPOINT_PATH)))
    model = load_model(checkpoint_fullpath)

    print("[INFO] old learning rate: {}".format(
        K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, base_alpha)
    print("[INFO] new learning rate: {}".format(
        K.get_value(model.optimizer.lr)))

# train the network
print("[INFO] Training model..........")
model.fit_generator(imagegen.flow(trainX, trainY, batch_size=64),
                    validation_data=(testX, testY), steps_per_epoch=len(trainX) // 64,
                    epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

# save the network to disk
print("[INFO] Serializing model into disk.........")
model.save(MODEL_PATH, overwrite=True)