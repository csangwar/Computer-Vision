# import the necessary packages
from keras.callbacks import Callback
import os


class EpochCheckpoint(Callback):
    def __init__(self, outputPath, every=1, startAt=0):
        # Call the parent constructor
        super(Callback, self).__init__()

        # Initialize the outputPath, Frequency of epoch to be saved,
        # and the current epoch value
        self.outputPath = outputPath
        self.every = every
        self.intEpoch = startAt

    def on_epoch_end(self, epoch, logs={}):
        # Check to see if the model should be serialized to the disk
        if (self.intEpoch + 1) % self.every == 0:
            p = os.path.sep.join([self.outputPath, "epoch_{}.hdf5".format(self.intEpoch +1)])
            self.model.save(p, overwrite=True)

        # Increment the internal epoch counter
        self.intEpoch += 1