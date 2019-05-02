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

import time
from sys import platform

import cv2
import d3dshot

from processFrame import *

from keyPressed import *
from steerTruck import *

# Initlize D3Dshot given user operating system
d = d3dshot.create(capture_output='numpy')


def drive():
    # Sorry Mac kiddos
    if platform == "win32":
        while True:
            # Remove last and now in final version
            last = time.time()
            screen = d.screenshot(region=(0, 35, 800, 600))
            newFrame = Frame(screen)
            newFrame = newFrame.transform()
            cv2.imshow('Jalopy', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            now = time.time()
            print("Windows took %f seconds" % (now - last))
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
