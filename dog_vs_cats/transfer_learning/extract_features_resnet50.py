# Coded by - u301671
# Import all the necessary packages
from catsanddogs_kaggle.dog_vs_cats.config import dogs_vs_cats_config as config
from keras.applications import ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from catsanddogs_kaggle.dog_vs_cats.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

# Construct the argument parser and parse the argumentsT
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
#                 help="path to input dataset")
# ap.add_argument("-o", "--output", required=True,
#                 help="path to output HDF5 file")
# ap.add_argument("-b", "--batch-size", type=int, default=16,
#                 help="batch-size of images to be passed through network")
# ap.add_argument("-s", "--buffer-size", type=int, default=1000,
#                 help="size of features extraction buffer")
# args = vars(ap.parse_args())

# Store the batch size in a variable
# bs = args["batch-size"]
bs = 16
OUTPUT_HDF5 = "./catsanddogs_kaggle/datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5"

# Input the images for processing, Shuffle them to allow easy training and testing
# splits via array slicing during training time
print("[INFO] Loading Images.....")
imagePaths = list(paths.list_images(config.IMAGES_PATH))
random.shuffle(imagePaths)   # Shuffle the images in the list

# extract the class labels from the image paths and then encode the labels
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# Load the RESNET 50 network
print("[INFO] loading network.....")
model = ResNet50(weights="imagenet", include_top=False)

# Store the features in HDF5 datset (Serialization)
# Initialize HDF5DatasetWriter, then store the class label names in the dataset
# output_dataset = HDF5DatasetWriter((len(imagePaths), 2048), args["output"],
#                                    dataKey="features", bufSize=args["buffer-size"])
output_dataset = HDF5DatasetWriter((len(imagePaths), 2048*7*7), OUTPUT_HDF5,
                                   dataKey="features", bufSize=1000)
output_dataset.storeClassLabels(le.classes_)

# Initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# Extracting features using CNN, loop over the images in batches
# loop over the images in batches
for i in np.arange(0, len(imagePaths), bs):
    # extract the batch of images and labels, then initialize the
    # list of actual images that will be passed through the network
    # for feature extraction
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []

    # Loop over the images and labels in current batch
    for (j, imagePath) in enumerate(batchPaths):
        # Load the image with resized value (224, 224)
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        # preprocess the image - 1) Expand the dimensions
        # 2) Subtracting the mean RGB pixel from Imagenet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image) # Default mode is caffe i.e. BGR mode

        # Add the image to the batch
        batchImages.append(image)

    # Pass the image through the network and use the output as our actual features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)
    features = features.reshape((features.shape[0], 2048 * 7 * 7))
    output_dataset.add(features, batchLabels)
    print("Value of i: ", i)
    pbar.update(i)

# Close the dataset
output_dataset.close()
pbar.finish()
