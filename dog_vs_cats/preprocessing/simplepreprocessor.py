# coded by u301671
# Import the necessary packages
import cv2


class SimpleProcessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.inter)