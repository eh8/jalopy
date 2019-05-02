import cv2
import numpy as np

class Frame(object):
    def __init__(self, img):
        self.img = img
        self.imgHeight = 200
        self.imgWidth = 800
        self.leftLane = 375
        self.rightLane = 425
        self.screen = np.float32(
            [[0, self.imgHeight+315], [self.imgWidth, self.imgHeight+315],
             [0, 315], [self.imgWidth, 315]])
        self.distortion = np.float32(
            [[self.leftLane, self.imgHeight], [self.rightLane, self.imgHeight],
             [0, 0], [self.imgWidth, 0]])

    def makeFrame(self, img):
        return Frame(img)

    def linearCombination(self, warped, s=1.0, m=0.0):
        # Does the work of actually adding contrast filter into the new image
        contrastImg = cv2.multiply(warped, np.array([s]))
        return cv2.add(contrastImg, np.array([m]))

    def contrastFilter(self, warped, s):
        # Increase constrast of image
        intensity = 127
        m = intensity*(1.0-s)
        return self.linearCombination(warped, s, m)

    def sharpenFilter(self, warped):
        # Increase grain of image
        sharpenImg = cv2.GaussianBlur(warped, (5, 5), 20.0)
        return cv2.addWeighted(warped, 2, sharpenImg, -1, 0)

    def transform(self):
        # Apply the warp mask onto the image
        # self.img = self.img[315:(315+self.imgHeight), 0:self.imgWidth]
        imgSize = (self.imgWidth, self.imgHeight)

        matrix = cv2.getPerspectiveTransform(self.screen, self.distortion)
        warped = cv2.warpPerspective(self.img, matrix, imgSize)
        warped = self.sharpenFilter(warped)
        warped = self.contrastFilter(warped, 1.0)
        return warped, matrix
