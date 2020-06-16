# Coded by-u301671
#################################
# Import the necessary packages #
#################################
import os
from catsanddogs_kaggle.dog_vs_cats.config import dogs_vs_cats_config as config
from catsanddogs_kaggle.dog_vs_cats.preprocessing.AspectAwarePreprocessor import AspectAwarePreprocessor
from catsanddogs_kaggle.dog_vs_cats.io.hdf5datasetwriter import HDF5DatasetWriter
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import progressbar
import json
import cv2
#
# for imagePath in paths.list_images("./catsanddogs_kaggle/datasets/kaggle_dogs_vs_cats/train"):
#     trainPaths = list(imagePath)
#     print(imagePath)
#     print(trainPaths)

trainPaths = list(paths.list_images(config.IMAGES_PATH))
# trainPaths[1]

trainLabels = [p.split(os.path.sep)[1].split(".")[0] for p in trainPaths]
# print(trainLabels)
# trainPaths[1].split(".")[0]
# trainPaths[1].split(os.path.sep)[1]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# Perform stratified sampling to build the Training and Testing datasets
split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES,
                         stratify=trainLabels, random_state=11)
(trainPaths, testPaths, trainLabels, testLabels) = split

# Stratified sampling to build the Validation datasets
split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_VAL_IMAGES,
                         stratify=trainLabels, random_state=11)
(trainPaths, valPaths, trainLabels, valLabels) = split

# Construct a consolidated list of all the files
datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5)]

# Initialize the AspectAwarePreprocessor
aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

# Loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
    # Create HDF5 writer
    print("INFO[]... Building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)

    # Initialize the progressbar
    widgets = ["BUILDING DATASET: ", progressbar.Percentage(), " ",
               progressbar.Bar(), progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # Loop over the image paths
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = aap.preprocess(image)

        if dType=="train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        writer.add([image], [label])
        pbar.update(i)

    pbar.finish()
    writer.close()

print("[INFO] Serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()






