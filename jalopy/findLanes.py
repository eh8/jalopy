import numpy as np
import cv2
from processFrame import Frame

# CODE CITATION: Inspired but personally implemeted version from a few sources:

# https://towardsdatascience.com/https-medium-com-priya-dwivedi-automatic-lane-detection-for-self-driving-cars-4f8b3dc0fb65
# AND
# https://medium.com/@cacheop/advanced-lane-detection-for-autonomous-cars-bff5390a360f


class Line():
    def __init__(self):
        self.bestFit = None
        self.reset()

    def reset(self):
        # flush all characteristics of the line
        # print("bad last attempt")
        self.detected = False
        self.lastAttempt = []
        self.currAttempt = [np.array([False])]
        # self.currAttempt = np.polyfit(yPoints, xPoints, 2)
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
            # print("error")
            lineFit = self.bestFit
            if initialAttempt:
                # print("resetting")
                self.reset()
            return lineFit


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
    margin = 40
    minpix = 30
    # Create empty lists to receive left and right lane pixel indices
    leftIndices = []
    rightIndices = []
    leftSensor = True
    rightSensor = True
    counter = 0

    # Load warped image
    warped = Frame(img)
    binaryWarp = warped.pipeline()

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

    for window in range(nwindows):
        # set dimensions of each window
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
            leftIndices.append(fairLeftIndices)
            rightIndices.append(fairRightIndices)
            counter += 1
        elif leftSensor:
            fairLeftIndices, leftcurX = slidingSensor(leftcurX, margin, minpix,
                                                      nonzerox, nonzeroy,
                                                      winYBottom, winYTop,
                                                      winMax, counter, 'left')
            leftIndices.append(fairLeftIndices)
        elif rightSensor:
            fairRightIndices, rightcurX = slidingSensor(rightcurX, margin,
                                                        minpix, nonzerox,
                                                        nonzeroy, winYBottom,
                                                        winYTop, winMax,
                                                        counter, 'right')
            rightIndices.append(fairRightIndices)
        else:
            break

    # Concatenate the arrays of indices
    leftIndices = np.concatenate(leftIndices)
    rightIndices = np.concatenate(rightIndices)

    # Extract left and right line pixel positions
    leftx = nonzerox[leftIndices]
    lefty = nonzeroy[leftIndices]
    rightx = nonzerox[rightIndices]
    righty = nonzeroy[rightIndices]

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
    # Fun fact: Mercedes Benz Actros is 6867 mm long

    warped = Frame(img)
    binaryWarp = warped.pipeline()

    # if we had lanes last time
    if leftLine.detected == False or rightLine.detected == False or\
       not np.any(leftLine) or not np.any(rightLine):
        initialLine(img, leftLine, rightLine)

    leftFit = leftLine.currAttempt
    rightFit = rightLine.currAttempt

    # Again, find the lane indicators
    nonzero = binaryWarp.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 40

    leftIndices = ((nonzerox > (leftFit[0]*(nonzeroy**2) +
                                leftFit[1]*nonzeroy + leftFit[2] - margin)) &
                   (nonzerox < (leftFit[0]*(nonzeroy**2) +
                                leftFit[1]*nonzeroy + leftFit[2] + margin)))

    rightIndices = ((nonzerox > (rightFit[0]*(nonzeroy**2) +
                                 rightFit[1]*nonzeroy + rightFit[2] - margin))
                    &
                    (nonzerox < (rightFit[0]*(nonzeroy**2) + rightFit[1]
                                 * nonzeroy + rightFit[2] + margin)))

    # Set the x and y values of points on each line
    leftx = nonzerox[leftIndices]
    lefty = nonzeroy[leftIndices]
    rightx = nonzerox[rightIndices]
    righty = nonzeroy[rightIndices]

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
    output[nonzeroy[leftIndices], nonzerox[leftIndices]] = [255, 0, 0]
    output[nonzeroy[rightIndices],
           nonzerox[rightIndices]] = [0, 0, 255]

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
        (1 + (2*leftFit[0]*expectedYCurve + leftFit[1])**2)**1.5) / np.absolute(2*leftFit[0])
    rightR = (
        (1 + (2*rightFit[0]*expectedYCurve + rightFit[1])**2)**1.5) / np.absolute(2*rightFit[0])

    leftFitCritical = np.polyfit(leftLine.allY,
                                 leftLine.allX, 2)
    rightFitCritical = np.polyfit(rightLine.allY,
                                  rightLine.allX, 2)

    # Calculate the new radii of curvature
    leftR = ((1 + (2*leftFitCritical[0]*expectedYCurve +
                   leftFitCritical[1])**2)**1.5) / np.absolute(2*leftFitCritical[0])

    rightR = ((1 + (2*rightFitCritical[0]*expectedYCurve +
                    rightFitCritical[1])**2)**1.5) / np.absolute(2*rightFitCritical[0])

    avgRadius = round(np.mean([leftR, rightR]), 0)

    radiusText = 'Radius = %s' % (avgRadius)

    middle_of_image = img.shape[1] / 2
    carPosition = middle_of_image

    leftLine_base = distFromCenter(leftFitCritical, img.shape[0])
    rightLine_base = distFromCenter(
        rightFitCritical, img.shape[0])
    laneCenter = (leftLine_base+rightLine_base)/2

    centerDeviation = laneCenter - carPosition
    if centerDeviation >= 0:
        centerText = '%s (left)' % (
            round(centerDeviation, 2))
    else:
        centerText = '%s (right)' % (
            round(centerDeviation, 2))

    # print(radiusText, centerText)

    if avgRadius > 50000 or abs(centerDeviation) > 100:
        # restart if the radius and the center deviation become something
        # ridiculous
        # leftLine = Line()
        # rightLine = Line()
        initialLine(img, leftLine, rightLine)
    

    # Invert the transform matrix from birds_eye (to later make the image back
    # to normal below)
    Minv = warped.createInverseMatrix()

    # Create an image to draw the lines on
    blankWarp = np.zeros_like(binaryWarp).astype(np.uint8)
    colorWarp = np.dstack((blankWarp, blankWarp, blankWarp))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pointsLeft = np.array([np.transpose(np.vstack([leftXFit, fity]))])
    pointsRight = np.array(
        [np.flipud(np.transpose(np.vstack([rightXFit, fity])))])
    pts = np.hstack((pointsLeft, pointsRight))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(colorWarp, np.int_([pts]), (26, 143, 227))

    newWarp = cv2.warpPerspective(
        colorWarp, Minv, (img.shape[1], img.shape[0]))

    # exitText = "Press Q to end Jalopy"
    
    # font = cv2.FONT_HERSHEY_DUPLEX
    # cv2.putText(newWarp, exitText, (10,50), font, 1,(255,255,255),2)

    # warp = warped.transform()[0]
    # warp = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)

    result = cv2.addWeighted(img, 1, newWarp, 0.6, 0)
    return result, avgRadius, centerDeviation


leftLine = Line()
rightLine = Line()


def processImage(image):
    global leftLine, rightLine
    # print(leftLine, rightLine)
    finishedImage, avgRadius, centerDeviation = drawLines(image,
                                                      leftLine, rightLine)

    # result = Frame(image)
    # result = result.pipeline()
    return finishedImage, avgRadius, centerDeviation
