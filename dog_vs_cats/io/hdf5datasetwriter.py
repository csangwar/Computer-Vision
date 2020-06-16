# Coding by - u301671
import os
import h5py


class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
        # If output path already exists then raise an exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'outputPath' already exists and cannot "
                             "be overwritten. Please delete it manually before "
                             "proceeding ahead", outputPath)
        # Open the HDF5 DB for writing and create 2 datsets:
        self.DB = h5py.File(outputPath, "w")
        # One to store the images/features
        self.data = self.DB.create_dataset(dataKey, dims, dtype="float")
        # Another to store the labels
        self.labels = self.DB.create_dataset("labels", (dims[0],), dtype="int")

        # Store the buffer size, then initialize the buffer itself along with the
        # index into the database
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        # Add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) > self.bufSize:
            self.flush()

    def flush(self):
        # Write the buffers in the disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx: i] = self.buffer["data"]
        self.labels[self.idx: i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        dt = h5py.string_dtype(encoding='utf-8')
        labelSet = self.DB.create_dataset("label_names", (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        # Check to see if there are any other entries in the buffer that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()
            # Close the dataset
            self.DB.close()

