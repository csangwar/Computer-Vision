# Coded by - u301671
# Project Configuration details --------------------->
#######################################################
# 1. Paths to the input imagess i.e. image directory  #
#######################################################
IMAGES_PATH = "./catsanddogs_kaggle/datasets/kaggle_dogs_vs_cats/train"
##############################################################################################
# 2. Total number of class labels and information of training, validation and testing splits #
##############################################################################################
NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

################################################################
# 3. Path to hdf5 datasets - training, validation and testing  #
################################################################
TRAIN_HDF5 = "./catsanddogs_kaggle/datasets/kaggle_dogs_vs_cats/hdf5/train.hdf5"
VAL_HDF5 = "./catsanddogs_kaggle/datasets/kaggle_dogs_vs_cats/hdf5/val.hdf5"
TEST_HDF5 = "./catsanddogs_kaggle/datasets/kaggle_dogs_vs_cats/hdf5/test.hdf5"

###############################################
# 4. Paths to output models, plots, logs etc. #
###############################################
MODEL_PATH = "./catsanddogs_kaggle/dog_vs_cats/dogs_vs_cats.model"
# Path to Dataset mean i.e. using MEAN SUBTRACTION METHOD for data Normalization
DATASET_MEAN = "./catsanddogs_kaggle/dog_vs_cats/output/dogs_vs_cats_mean.json"
# Path to output plots, classification reports etc.
OUTPUT_PATH = "./catsanddogs_kaggle/dog_vs_cats/output"
#
