# Eric Cheng

import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
from PIL import Image

import d3dshot
from findLanes import *
from keyPressed import *
from mss import mss as sct
from steerTruck import *

# Initlize D3Dshot
try:
    d = d3dshot.create(capture_output='numpy')
except:
    print("Not Windows")


def config():
    # User settings
    binSize = 1
    degree = np.pi / 180
    minLineLength = 40
    maxLineGap = 5
    return binSize, degree, minLineLength, maxLineGap
    # User should not modify from this point onwards


def initParams(ran):
    global right_fit_p, left_fit_p, n_count, RANGE, MIN_POINTS
    right_fit_p = np.zeros(POL_ORD+1)
    left_fit_p = np.zeros(POL_ORD+1)
    n_count = 0
    RANGE = ran
    MIN_POINTS = 25-15*ran


def run():
    M, Minv = create_M()

    while True:
        last = time.time()
        if sys.platform.startswith('win'):
            screen = d.screenshot(region=(0, 35, 800, 600))
            screen = transform(screen, M)
            os = 'Windows'
        else:
            mon = {'top': 35, 'left': 0, 'width': 800, 'height': 600}
            sct = mss()
            sct.get_pixels(mon)
            os = 'Frame'
            img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
            screen = np.array(img)
        cv2.imshow('test', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        now = time.time()
        print("%s took %f seconds" % (os, now - last))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        # plt.imshow(screen)
        # if plt.show() & 0xFF == ord('s'):
        #     break


def main():
    initParams(0.85)
    run()


if __name__ == '__main__':
    main()
