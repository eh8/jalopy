# Eric Cheng
import cv2
import numpy as np

# Testing an image
image = cv2.imread('test_image.jpg')
laneImage = np.copy(image)
grayImage = cv2.cvtColor(laneImage, cv2.COLOR_RGB2GRAY)
gaussianImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
cv2.imshow('Jalopy', gaussianImage)
cv2.waitKey(0)

