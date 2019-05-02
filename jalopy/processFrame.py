import cv2
import numpy as np


class Frame(object):
    def __init__(self, img):
        self.img = img
        self.imgHeight = 600
        self.imgWidth = 800
        self.leftLane = 360
        self.rightLane = 420
        self.screen = np.float32(
            [[430, 320],
             [800, 550],
             [0, 550],
             [370, 320],])
        self.offset = 200
        self.distortion = np.float32([[600, 0],
                                      [600, 600],
                                      [200, 600],
                                      [200, 0]])

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
        warped = self.contrastFilter(warped, 1.3)
        return warped, matrix

    def pipeline(self):
        s_thresh = 150, 255
        sx_thresh = 10, 100
        R_thresh = 100, 255
        sobel_kernel = 3

        warp = Frame(self.img)
        warp = self.transform()[0]
        # warp = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)
        R = warp[:, :, 0]

        hls = cv2.cvtColor(warp, cv2.COLOR_RGB2HLS).astype(np.float)
        # split into three separate channels for discrimination
        hChannel = hls[:, :, 0]
        lChannel = hls[:, :, 1]
        sChannel = hls[:, :, 2]

        # Generate sobel edge mask - this filter checks for edges
        sobelx = cv2.Sobel(lChannel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Generate sobel binary filter - all bits between these two thresholds
        # are selected
        sxBinary = np.zeros_like(scaled_sobelx)
        sxBinary[(scaled_sobelx >= sx_thresh[0])
                 & (scaled_sobelx <= sx_thresh[1])] = 1

        # This filter checks for brightness
        rBinary = np.zeros_like(R)
        rBinary[(R >= R_thresh[0]) & (R <= R_thresh[1])] = 1

        # This filter kinda sucks because it only tracks color
        # sBinary = np.zeros_like(sChannel)
        # sBinary[(sChannel >= s_thresh[0]) & (sChannel <= s_thresh[1])] = 1

        compositeImage = np.zeros_like(rBinary)
        compositeImage[((sxBinary == 1) & (rBinary == 1))] = 1
        compositeImage = 255 * compositeImage
        return compositeImage

    def createInverseMatrix(self):
        inverseMatrix = cv2.getPerspectiveTransform(self.distortion,
                                                    self.screen)
        return inverseMatrix
