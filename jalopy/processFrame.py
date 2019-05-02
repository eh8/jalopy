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

    def pipeline(self, s_thresh=(125, 255), sx_thresh=(10, 100),
                 R_thresh=(200, 255), sobel_kernel=3):

        warp, matrix = self.transform()
        R = warp[:, :, 0]

        hls = cv2.cvtColor(warp, cv2.COLOR_RGB2HLS).astype(np.float)
        lChannel = hls[:, :, 1]
        sChannel = hls[:, :, 2]

        sobelx = cv2.Sobel(lChannel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        sxbinary = np.zeros_like(scaled_sobelx)
        sxbinary[(scaled_sobelx >= sx_thresh[0])
                 & (scaled_sobelx <= sx_thresh[1])] = 1

        rBinary = np.zeros_like(R)
        rBinary[(R >= R_thresh[0]) & (R <= R_thresh[1])] = 1

        sBinary = np.zeros_like(sChannel)
        sBinary[(sChannel >= s_thresh[0]) & (sChannel <= s_thresh[1])] = 1

        compositeImage = np.zeros_like(sxbinary)
        compositeImage[((sBinary == 1) & (sxbinary == 1))
                       | ((sxbinary == 1) & (rBinary == 1))
                       | ((sBinary == 1) & (rBinary == 1))] = 1

        return compositeImage
