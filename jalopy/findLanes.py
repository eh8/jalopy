import numpy as np
import cv2
from processFrame import *

# CODE CITATION: Inspired but heavily modified version by
# https://towardsdatascience.com/https-medium-com-priya-dwivedi-automatic-lane-detection-for-self-driving-cars-4f8b3dc0fb65
# AND
# https://medium.com/@cacheop/advanced-lane-detection-for-autonomous-cars-bff5390a360f


class Line():
    def __init__(self):
        self.bestFit = None
        self.reset()

    def reset(self):
        # flush all characteristics of the line
        self.detected = False
        self.lastAttempt = []
        self.currAttempt = [np.array([False])]
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.allX = None
        self.allY = None
        self.counter = 0

    def lineFit(self, xPoints, yPoints, initialAttempt=True):
        try:
            n = 5
            self.currAttempt = np.polyfit(yPoints, xPoints, 2)
            self.allX = xPoints
            self.allY = yPoints
            self.lastAttempt.append(self.currAttempt)
            if len(self.lastAttempt) > 1:
                self.diffs = (
                    self.lastAttempt[-2] - self.lastAttempt[-1]) /\
                    self.lastAttempt[-2]
            self.lastAttempt = self.lastAttempt[-n:]
            self.bestFit = np.mean(self.lastAttempt, axis=0)
            lineFit = self.currAttempt
            self.detected = True
            self.counter = 0

            return lineFit

        except (TypeError, np.linalg.LinAlgError):
            lineFit = self.bestFit
            if initialAttempt:
                self.reset()
            return lineFit


def pipeline(img, s_thresh=(125, 255), sx_thresh=(10, 100),
             R_thresh=(200, 255), sobel_kernel=3):

    R = img[:, :, 0]

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
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


def slidingSensor(curX, margin, minpix, nonzerox, nonzeroy,
                  winYBottom, winYTop, winMax, counter, side):

    winXBottom = curX - margin
    winXTop = curX + margin
    fairPoints = ((nonzeroy >= winYBottom) & (nonzeroy < winYTop)
                  & (nonzerox >= winXBottom)
                  & (nonzerox < winXTop)).nonzero()[0]
    if len(fairPoints) > minpix:
        curX = np.int(np.mean(nonzerox[fairPoints]))
    if counter >= 5:
        if winXTop > winMax or winXBottom < 0:
            if side == 'left':
                leftSensor = False
            else:
                rightSensor = False

    return fairPoints, curX


def initialLine(img, leftLine, rightLine):
    # number of sensors
    nwindows = 35
    margin = 100
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    leftLaneIndices = []
    rightLaneIndices = []
    leftSensor = True
    rightSensor = True
    counter = 0

    # Load warped image
    warped = Frame(img)
    binaryWarp, matrix = warped.transform()

    histogram = np.sum(
        binaryWarp[int(binaryWarp.shape[0]/2):, :], axis=0)

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(binaryWarp.shape[0]/nwindows)

    nonzero = binaryWarp.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftcurX = leftx_base
    rightcurX = rightx_base

    # Step through the windows one by one
    for window in range(nwindows):
        winYBottom = binaryWarp.shape[0] - (window+1)*window_height
        winYTop = binaryWarp.shape[0] - window*window_height
        winMax = binaryWarp.shape[1]
        if leftSensor and rightSensor:
            fairLeftIndices, leftcurX = slidingSensor(leftcurX, margin, minpix,
                                                      nonzerox, nonzeroy,
                                                      winYBottom, winYTop,
                                                      winMax, counter, 'left')
            fairRightIndices, rightcurX = slidingSensor(rightcurX, margin,
                                                        minpix, nonzerox,
                                                        nonzeroy,
                                                        winYBottom, winYTop,
                                                        winMax, counter, 'right')
            leftLaneIndices.append(fairLeftIndices)
            rightLaneIndices.append(fairRightIndices)
            counter += 1
        elif leftSensor:
            fairLeftIndices, leftcurX = slidingSensor(leftcurX, margin, minpix,
                                                      nonzerox, nonzeroy,
                                                      winYBottom, winYTop,
                                                      winMax, counter, 'left')
            leftLaneIndices.append(fairLeftIndices)
        elif rightSensor:
            fairRightIndices, rightcurX = slidingSensor(rightcurX, margin,
                                                        minpix, nonzerox,
                                                        nonzeroy, winYBottom,
                                                        winYTop, winMax,
                                                        counter, 'right')
            rightLaneIndices.append(fairRightIndices)
        else:
            break

    # Concatenate the arrays of indices
    leftLaneIndices = np.concatenate(leftLaneIndices)
    rightLaneIndices = np.concatenate(rightLaneIndices)

    # Extract left and right line pixel positions
    leftx = nonzerox[leftLaneIndices]
    lefty = nonzeroy[leftLaneIndices]
    rightx = nonzerox[rightLaneIndices]
    righty = nonzeroy[rightLaneIndices]

    # Fit a second order polynomial to each line
    leftFit = leftLine.lineFit(leftx, lefty, True)
    rightFit = rightLine.lineFit(rightx, righty, True)


def distFromCenter(line, val):
    a = line[0]
    b = line[1]
    c = line[2]
    formula = (a*val**2)+(b*val)+c
    return formula


def drawLines(img, leftLine, rightLine):
    # rough pixel to meter conversion
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    warped = Frame(img)
    binaryWarp, matrix = warped.transform()

    # if we had lanes last time
    if leftLine.detected == False or rightLine.detected == False:
        initialLine(img, leftLine, rightLine)

    leftFit = leftLine.currAttempt
    rightFit = rightLine.currAttempt

    # Again, find the lane indicators
    nonzero = binaryWarp.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    leftLaneIndices = ((nonzerox > (leftFit[0]*(nonzeroy**2) +
                                    leftFit[1]*nonzeroy + leftFit[2] - margin)) &
                       (nonzerox < (leftFit[0]*(nonzeroy**2) +
                                    leftFit[1]*nonzeroy + leftFit[2] + margin)))

    rightLaneIndices = ((nonzerox > (rightFit[0]*(nonzeroy**2) +
                                     rightFit[1]*nonzeroy + rightFit[2] - margin))
                        &
                        (nonzerox < (rightFit[0]*(nonzeroy**2) + rightFit[1]
                                     * nonzeroy + rightFit[2] + margin)))

    # Set the x and y values of points on each line
    leftx = nonzerox[leftLaneIndices]
    lefty = nonzeroy[leftLaneIndices]
    rightx = nonzerox[rightLaneIndices]
    righty = nonzeroy[rightLaneIndices]

    # Fit a second order polynomial to each again.
    leftFit = leftLine.lineFit(leftx, lefty, False)
    rightFit = rightLine.lineFit(rightx, righty, False)

    # Generate x and y values for plotting
    fity = np.linspace(0, binaryWarp.shape[0]-1, binaryWarp.shape[0])
    leftXFit = leftFit[0]*fity**2 + leftFit[1]*fity + leftFit[2]
    rightXFit = rightFit[0]*fity**2 + rightFit[1]*fity + rightFit[2]

    # Create an image to draw on and an image to show the selection window
    output = np.dstack((binaryWarp, binaryWarp, binaryWarp))*255
    window_img = np.zeros_like(output)

    # Color in left and right line pixels
    output[nonzeroy[leftLaneIndices], nonzerox[leftLaneIndices]] = [255, 0, 0]
    output[nonzeroy[rightLaneIndices],
           nonzerox[rightLaneIndices]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    leftLine_window1 = np.array(
        [np.transpose(np.vstack([leftXFit-margin, fity]))])
    leftLine_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([leftXFit+margin, fity])))])
    leftLine_pts = np.hstack((leftLine_window1, leftLine_window2))
    rightLineWindow1 = np.array(
        [np.transpose(np.vstack([rightXFit-margin, fity]))])
    rightLineWindow2 = np.array(
        [np.flipud(np.transpose(np.vstack([rightXFit+margin, fity])))])
    rightLine_pts = np.hstack((rightLineWindow1, rightLineWindow2))

    expectedYCurve = np.max(fity)
    leftR = (
        (1 + (2*leftFit[0]*expectedYCurve + leftFit[1])**2)**1.5) /
    np.absolute(2*leftFit[0])
    rightR = (
        (1 + (2*rightFit[0]*expectedYCurve + rightFit[1])**2)**1.5) /
    np.absolute(2*rightFit[0])

    leftFitCritical = np.polyfit(leftLine.allY*ym_per_pix,
                                 leftLine.allX*xm_per_pix, 2)
    rightFitCritical = np.polyfit(rightLine.allY*ym_per_pix,
                                  rightLine.allX*xm_per_pix, 2)

    # Calculate the new radii of curvature
    leftR = ((1 + (2*leftFitCritical[0]*expectedYCurve*ym_per_pix +
                   leftFitCritical[1])**2)**1.5) /
    np.absolute(2*leftFitCritical[0])

    rightR = ((1 + (2*rightFitCritical[0]*expectedYCurve*ym_per_pix +
                    rightFitCritical[1])**2)**1.5) /
    np.absolute(2*rightFitCritical[0])

    avg_rad = round(np.mean([leftR, rightR]), 0)
    rad_text = 'Radius of Curvature = {}(m)'.format(avg_rad)

    middle_of_image = img.shape[1] / 2
    carPosition = middle_of_image * xm_per_pix

    leftLine_base = distFromCenter(leftFitCritical, img.shape[0] * ym_per_pix)
    rightLine_base = distFromCenter(
        rightFitCritical, img.shape[0] * ym_per_pix)
    lane_mid = (leftLine_base+rightLine_base)/2

    centerDeviation = lane_mid - carPosition
    if centerDeviation >= 0:
        center_text = '{} meters left of center'.format(
            round(centerDeviation, 2))
    else:
        center_text = '{} meters right of center'.format(
            round(-centerDeviation, 2))

    # Invert the transform matrix from birds_eye (to later make the image back
    #   to normal below)
    Minv = np.linalg.inv(matrix)

    # Create an image to draw the lines on
    blankWarp = np.zeros_like(binaryWarp).astype(np.uint8)
    colorWarp = np.dstack((blankWarp, blankWarp, blankWarp))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([leftXFit, fity]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([rightXFit, fity])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(colorWarp, np.int_([pts]), (0, 255, 0))

    newwarp = cv2.warpPerspective(
        colorWarp, Minv, (img.shape[1], img.shape[0]))

    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result


def processImage(image):
    # leftLine = Line()
    # rightLine = Line()

    # result = drawLines(image, leftLine, rightLine)
    result = pipeline(image)

    return result
