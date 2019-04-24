# Eric Cheng

# Imaging libraries (mss is faster than standard PIL)
from mss import mss
from PIL import Image

# Controlling game state
from keyPressed import ReleaseKey, PressKey, W, A, S, D
from steerTruck import *

# lane-finding
# from findLanes import *

# Calculations
from statistics import mean

# Computer vision and auxiliary packages
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
import cv2

# Debugging
import time
import matplotlib.pyplot as plt
import pprint
import pyautogui
pp = pprint.PrettyPrinter()

############################### User settings ##################################


def config():
    binSize = 1
    degree = np.pi / 180
    minLineLength = 100
    maxLineGap = 5
    return binSize, degree, minLineLength, maxLineGap

######################### User should not modify ###############################


def drawLines(img, lines, color=[0, 255, 255], thickness=3):
    # if this fails, go with some default line
    try:

        # finds the maximum y value for a lane marker
        # (since we cannot assume the horizon will always be at the same point.)

        ys = []
        for i in lines:
            for ii in i:
                ys += [ii[1], ii[3]]
        min_y = min(ys)
        max_y = 600
        new_lines = []
        line_dict = {}

        for idx, i in enumerate(lines):
            for xyxy in i:
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Used to calculate the definition of a line, given two sets of coords.
                x_coords = (xyxy[0], xyxy[2])
                y_coords = (xyxy[1], xyxy[3])
                A = vstack([x_coords, ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]

                # Calculating our new, and improved, xs
                x1 = (min_y-b) / m
                x2 = (max_y-b) / m

                line_dict[idx] = [m, b, [int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]

            if len(final_lanes) == 0:
                final_lanes[m] = [[m, b, line]]

            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m, b, line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [[m, b, line]]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(),
                           key=lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s))

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2], lane1_id, lane2_id
    except Exception as e:
        print(str(e))


def roi(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked


def processImage(originalImage):
    (binSize, degree, length, gap) = config()
    processedImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    processedImage = cv2.Canny(processedImage, threshold1=200, threshold2=300)
    processedImage = cv2.GaussianBlur(processedImage, (5, 5), 0)
    vertices = np.array([[0, 600], [0, 450], [300, 300], [500, 300], [800, 450],
                         [800, 600]])
    processedImage = roi(processedImage, [vertices])
    lines = cv2.HoughLinesP(processedImage, binSize, degree, 180,
                            np.array([]), length, gap)
    m1 = 0
    m2 = 0
    try:
        l1, l2, m1, m2 = drawLines(originalImage, lines)
        cv2.line(originalImage, (l1[0], l1[1]),
                 (l1[2], l1[3]), [0, 255, 0], 30)
        cv2.line(originalImage, (l2[0], l2[1]),
                 (l2[2], l2[3]), [0, 255, 0], 30)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processedImage, (coords[0], coords[1]), (coords[2], coords[3]), [
                         255, 0, 0], 3)

            except Exception as e:
                print(str(e))
    except Exception as e:
        pass

    return processedImage, originalImage, m1, m2


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
        newScreen, originalImage, m1, m2 = processImage(screen)

        if m1 < 0 and m2 < 0:
            right()
            print('right')
        elif m1 > 0 and m2 > 0:
            left()
            print('left')
        else:
            straight()
            print('straight')

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
