import d3dshot
import numpy as np
import cv2
import time
import sys
from mss import mss
from PIL import Image

d = d3dshot.create(capture_output='numpy')

if sys.platform.startswith('win'):
    while True:
        last = time.time()
        screen = d.screenshot(region=(0, 35, 800, 600))
        cv2.imshow('test', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        now = time.time()
        print("Frame took %f seconds" % (now - last))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
else:
    mon = {'top': 35, 'left': 0, 'width': 800, 'height': 600}

    sct = mss()

    while True:
        sct.get_pixels(mon)
        img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
        screen = np.array(img)
        cv2.imshow('test', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

# undistort

# import glob

x_cor = 9  # Number of corners to find
y_cor = 6
# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((y_cor*x_cor, 3), np.float32)
objp[:, :2] = np.mgrid[0:x_cor, 0:y_cor].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.
# Make a list of paths to calibration images
images = glob.glob('camera_cal/calibration*.jpg')
# Step through the list and search for chessboard corners
corners_not_found = []
# Calibration images in which opencv failed to find corners

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Conver to grayscale
    ret, corners = cv2.findChessboardCorners(
        gray, (x_cor, y_cor), None)  # Find the chessboard corners
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (x_cor, y_cor), corners, ret)
    else:
        corners_not_found.append(fname)

img = cv2.imread('camera_cal/calibration5.jpg')
img_size = (img.shape[1], img.shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None)


def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

# Unix support
    else:
            mon = {'top': 35, 'left': 0, 'width': 800, 'height': 600}
            sct = mss()
            sct.get_pixels(mon)
            os = "Unix"
            img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
            screen = np.array(img)
