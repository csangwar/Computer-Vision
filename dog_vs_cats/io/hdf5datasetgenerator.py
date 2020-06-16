# Coded by - u301671
# Import the necessary packages

from keras.utils import np_utils
import numpy as np
import h5py


class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, classes=2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0
        # arange (start, end, steps)
        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]

                # check to see if the labels should be binarized
                if self.binarize:
                    labels = np_utils.to_categorical(labels, num_classes=self.classes)

                # Check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # initialize the list of processed images
                    procImages = []

                    # Loop over the images
                    for image in images:
                        # Loop over the preprocessors and apply each to image
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        # Update the list of processed images
                        procImages.append(image)

                    # Update the images array as processed images
                    images = np.array(procImages)

                # If the data augmentor exists then apply it
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels,
                                                          batch_size=self.batchSize))
                # Yield a tuple of images and labels
                yield (images, labels)

        # increment the total number of epochs
        epochs += 1

    def close(self):
        self.db.close()