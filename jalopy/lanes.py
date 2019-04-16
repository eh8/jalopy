# Eric Cheng
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pprint

pp = pprint.PrettyPrinter()


def coordinates(image, parameters):
    slope, intercept = parameters
    offset = 0.5
    y1 = image.shape[0]
    y2 = int(y1 * offset)
    # y = mx + b ->
    # x = (y - b)/m
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def avgMB(image, lines):
    left = []
    right = []
    degree = 1
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), degree)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))
    leftAvg = np.average(left, axis=0)
    rightAvg = np.average(right, axis=0)
    leftLine = coordinates(image, leftAvg)
    rightLine = coordinates(image, rightAvg)
    return np.array([leftLine, rightLine])


def cannify(image):
    # Pre-process image by applying grayscale, gaussian blur,
    # and derivative filter
    alpha = 2
    beta = 20
    mask = np.zeros(image.shape, image.dtype)
    contrast = cv2.addWeighted(image, alpha, mask, 0, beta)
    gray = cv2.cvtColor(contrast, cv2.COLOR_RGB2GRAY)
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(gaussian, 0, 150)
    return canny


def regionOfInterest(image):
    # numpy array is (m, n, l)
    height = image.shape[0]

    # # IRL
    # lowerLeft = (200, height)
    # center = (1100, height)
    # lowerRight = (550, 250)

    # Euro Truck Turns 
    lowerLeft = (540, 480)
    center = (600, 250)
    lowerRight = (900, 510)

    # # Euro Truck Straight
    # lowerLeft = (500, 450)
    # center = (650, 200)
    # lowerRight = (950, 450)

    region = [lowerLeft, center, lowerRight]
    poly = np.array([region])
    # completely black image
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, poly, 255)
    maskedImage = cv2.bitwise_and(image, mask)
    # pp.pprint(mask)
    return maskedImage


def houghFit(image):
    binSize = 2
    degrees = 180
    dPrecision = np.pi/degrees
    # I found that 100 works best, will fine tune later
    threshold = 90
    minLL = 40
    maxLG = 2
    lines = cv2.HoughLinesP(image, binSize, dPrecision, threshold,
                            minLineLength=minLL, maxLineGap=maxLG)
    return lines


def displayLines(image, lines):
    lineImage = np.zeros_like(image)
    color = (0, 255, 0)
    thickness = 5
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lineImage, (x1, y1), (x2, y2), color, thickness)
    return lineImage


def main():
    cap = cv2.VideoCapture('./euro.mp4')
    while cap.isOpened():
        # image = cv2.imread('test_image.jpg')
        _, frame = cap.read()
        laneImage = np.copy(frame)
        cannyImage = cannify(laneImage)
        maskImage = regionOfInterest(cannyImage)
        lines = houghFit(maskImage)
        # averageLines = avgMB(laneImage, lines)
        lineImage = displayLines(laneImage, lines)
        superImage = cv2.addWeighted(laneImage, 0.8, lineImage, 1, 1)
        cv2.imshow('Jalopy', superImage)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        # plt.imshow(laneImage)
        # if plt.show() & 0xFF == ord('s'):
        #     break
    cap.release()
    cv2.destroyAllWindows()


main()
