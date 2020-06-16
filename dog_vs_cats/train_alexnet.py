# Coded by u301671
# Import necessary packages
import matplotlib
matplotlib.use("Agg")

from catsanddogs_kaggle.dog_vs_cats.config import dogs_vs_cats_config as config
from catsanddogs_kaggle.dog_vs_cats.preprocessing.AspectAwarePreprocessor import AspectAwarePreprocessor
from catsanddogs_kaggle.dog_vs_cats.preprocessing.croppreprocessor import CropPreprocessor
from catsanddogs_kaggle.dog_vs_cats.preprocessing.meanpreprocessor import MeanPreprocessor
from catsanddogs_kaggle.dog_vs_cats.preprocessing.patchpreprocessor import PatchPreprocessor
from catsanddogs_kaggle.dog_vs_cats.preprocessing.simplepreprocessor import SimpleProcessor
from catsanddogs_kaggle.dog_vs_cats.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from catsanddogs_kaggle.dog_vs_cats.callbacks.trainingmonitor import TrainingMonitor
from catsanddogs_kaggle.dog_vs_cats.io.hdf5datasetgenerator import HDF5DatasetGenerator
from catsanddogs_kaggle.dog_vs_cats.neural_nets.conv.alexnet import Alexnet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

# Training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimpleProcessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(dbPath=config.TRAIN_HDF5, batchSize=128, aug=aug, preprocessors=[pp, mp, iap],
                                classes=2)
valGen = HDF5DatasetGenerator(dbPath=config.VAL_HDF5, batchSize=128, preprocessors=[sp, mp, iap],
                              classes=2)

# Initialize the optimizer
print("[INFO] Compiling model.....")
opt = Adam(lr=1e-3)
model = Alexnet.build(width=227, height=227, depth=3,
                      classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

#Train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 128,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 128,
    epochs=75,
    max_queue_size=128*2,
    callbacks=callbacks, verbose=1)

# Save the model to file
print("[INFO] serializing model....")
model.save(config.MODEL_PATH, overwrite=True)

# Close the HDF5 datasets
trainGen.close()
valGen.close()