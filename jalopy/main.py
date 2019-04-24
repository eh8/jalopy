# Eric Cheng

# Imaging libraries (mss is faster than standard PIL)
from mss import mss
from PIL import Image

# Controlling game state
from keyPressed import ReleaseKey, PressKey, W, A, S, D

# lane-finding
# from findLanes import *
from findLanes import *

# Actual OpenCV
import numpy as np
import cv2

# Debugging
import time
import matplotlib.pyplot as plt
import pprint
import pyautogui
pp = pprint.PrettyPrinter()


def config():
	binSize = 1
	degree = np.pi / 180
	maxLineLength = 100
	maxLineGap = 5
	return binSize, degree, maxLineLength, maxLineGap


def roi(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked


def processImage(originalImage):
	binSize, degree, maxLineLength, maxLineGap = config()
    processedImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    processedImage = cv2.Canny(
        processedImage, threshold1=200, threshold2=300)
    vertices = np.array([[0, 600], [0, 450], [300, 300], [
                        500, 300], [800, 450], [800, 600]])
    processedImage = roi(processedImage, [vertices])
	lines = cv2.HougeLinesP(processedImage, binSize, degree, length, gap)
    return processedImage


def countDown():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)


def boot():
    while True:
        roi = {'top': 35, 'left': 0, 'width': 800, 'height': 600}
        screenshot = mss()
        screenshot.get_pixels(roi)
        screen = Image.frombytes('RGB', (screenshot.width, screenshot.height),
                                 screenshot.image)
        screen = np.array(screen)
        newScreen = processImage(screen)

        # print('Down')
        # PressKey(W)
        # time.sleep(3)
        # print('Up')
        # PressKey(W)

        cv2.imshow('Jalopy', cv2.cvtColor(newScreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        # plt.imshow(newScreen)
        # if plt.show() & 0xFF == ord('s'):
        #     break


def main():
    countDown()
    boot()


if __name__ == '__main__':
    main()
