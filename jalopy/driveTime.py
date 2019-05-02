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

from sys import platform

import cv2
import d3dshot

from processFrame import Frame
from findLanes import *
# from full_pipeline import *
from keyPressed import *
from steerTruck import *
# from drawLanes import draw_lane
import matplotlib.pyplot as plt

d = d3dshot.create(capture_output='numpy')

lastMoves = []
cenDivs = [0]
derivatives = []


def drive():
    # Sorry Mac kiddos
    if platform == "win32":
        isActive = True
        while True:
            # Remove last and now in final version
            screen = d.screenshot(region=(0, 35, 800, 635))
            if isActive:
                screen, avgRadius, centerDeviation = processImage(screen)

                lastDiv = cenDivs[0]
                cenDivs.append(centerDeviation)
                derivativeBias = cenDivs[1] - cenDivs[0]
                derivatives.append(derivativeBias)

                steeringThreshold = (-40, 40)

                # restrict memory to 8 ticks

                if len(lastMoves) > 10:
                    if ((np.array(derivatives) < -2).sum() == np.array(derivatives).size).astype(np.int) == 1 and lastMoves.count(3) < 2:
                        # print("emergency left", derivatives, derivativeBias)
                        lastMoves.append(3)
                        left()
                        slow()

                    elif ((np.array(derivatives) > 2).sum() == np.array(derivatives).size).astype(np.int) == 1 and lastMoves.count(4) < 2:
                        # print("emergency right", derivatives, derivativeBias)
                        lastMoves.append(4)
                        right()
                        slow()

                    elif centerDeviation >= steeringThreshold[1] and\
                            lastMoves.count(2) < 2:
                        # print("turning right", centerDeviation)
                        lastMoves.append(2)
                        right()

                    elif centerDeviation <= steeringThreshold[0] and\
                            lastMoves.count(1) < 2:
                        # print("turning left", centerDeviation)
                        lastMoves.append(1)
                        left()

                    else:
                        # print("goinwg straight", centerDeviation)
                        lastMoves.append(0)
                        straight()
                    lastMoves.pop(0)
                else:
                    straight()
                    lastMoves.append(0)

                if len(cenDivs) > 2:
                    cenDivs.pop(0)

                # restrict memory to 6 ticks

                if len(derivatives) > 10:
                    derivatives.pop(0)

                # print(lastMoves, avgRadius, centerDeviation, derivativeBias)

                # cv2.imshow('Jalopy', lanes)
                cv2.imshow('Jalopy', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            # plt.imshow(screen)
            # if plt.show() & 0xFF == ord('s'):
            #     break

            # Pause OpenCv
            if cv2.waitKey(10) & 0xFF == ord('p'):
                # print("pause!")
                pauseMenu = cv2.imread("Pause.png")
                result = cv2.addWeighted(pauseMenu, 1, screen, 0.9, 0)
                cv2.imshow('Jalopy', result)
                isActive = False

            # Continue OpenCV
            if cv2.waitKey(10) & 0xFF == ord('c'):
                # print("unpause!")
                isActive = True


def main():
    drive()


if __name__ == '__main__':
    main()
