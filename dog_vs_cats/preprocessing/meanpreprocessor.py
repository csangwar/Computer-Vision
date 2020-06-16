# coded by - u301671
##############################################
# Image Normalization using Mean subtraction #
##############################################
import cv2


class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        (B, G, R) = cv2.split(image.astype("float32"))
        # Subtract mean from each Channel's pixels
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean
        # Since Opencv accepts the images as B, G, R and hence merging in same order
        return cv2.merge([B, G, R])