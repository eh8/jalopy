#        _       _
#       | |     | |
#       | | __ _| | ___  _ __  _   _
#   _   | |/ _` | |/ _ \| '_ \| | | |
#  | |__| | (_| | | (_) | |_) | |_| |
#   \____/ \__,_|_|\___/| .__/ \__, |
#                       | |     __/ |
#                       |_|    |___/
#
# A self-driving system for Euro Truck Simulator 2
# Eric Cheng
# 15-112 TP

import sys
import time

import cv2
import numpy as np
from PIL import Image

import d3dshot
from processFrame import *
from keyPressed import *
from mss import mss
from steerTruck import *

# Initlize D3Dshot given user operating system
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


def drive():
    # Generate distortion matrix for transformation
    M, Minv = create_M()

    while True:
        last = time.time()
        if sys.platform.startswith('win'):
            screen = d.screenshot(region=(0, 35, 800, 600))
            os = 'Windows'
        else:
            mon = {'top': 35, 'left': 0, 'width': 800, 'height': 600}
            sct = mss()
            sct.get_pixels(mon)
            os = "Unix"
            img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
            screen = np.array(img)
        screen = transform(screen, M)
        cv2.imshow('Jalopy', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        now = time.time()
        print("%s took %f seconds" % (os, now - last))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        # plt.imshow(screen)
        # if plt.show() & 0xFF == ord('s'):
        #     break


def main():
    drive()


if __name__ == '__main__':
    main()
