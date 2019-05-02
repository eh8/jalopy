from processFrame import Frame
from findLanes import *
import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import mean_squared_error


IMAGE_H = 200
IMAGE_W = 800

WINDOW_SIZE = 15  # Half of the sensor span
DEV = 7  # Maximum of the point deviation from the sensor center
SPEED = 2 / IMAGE_H  # Pixels shift per frame
POL_ORD = 2  # Default polinomial order
RANGE = 0.0  # Fraction of the image to skip

# for find
right_fit_p = np.zeros(POL_ORD+1)
left_fit_p = np.zeros(POL_ORD+1)
r_len = 0.1
l_len = 0.1
lane_w_p = 90

MIN = 60  # Minimal line separation (in px)
MAX = 95  # Maximal line separation (in px)
MIN_POINTS = 10  # Minimal points to consider a line
MAX_N = 5  # Maximal frames without line detected to use previous frame
n_count = 0  # Frame counter
r_n = 0  # Number of frames with unsuccessful line detection
l_n = 0

LEFT_LANE = 375
RIGHT_LANE = 425

screen = np.float32(
    [[0, IMAGE_H+315], [IMAGE_W, IMAGE_H+315],
     [0, 315], [IMAGE_W, 315]])
distortion = np.float32(
    [[LEFT_LANE, IMAGE_H], [RIGHT_LANE, IMAGE_H],
     [0, 0], [IMAGE_W, 0]])

M = cv2.getPerspectiveTransform(screen, distortion)
Minv = cv2.getPerspectiveTransform(distortion, screen)

RANGE = 0.0


def s_thres(img, thresh_min=25, thresh_max=255):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > thresh_min) & (s_channel <= thresh_max)] = 1
    return s_binary


def pol_shift(pol, h):
    pol_ord = len(pol)-1  # Determinate degree of the polynomial
    if pol_ord == 3:
        pol0 = pol[0]
        pol1 = pol[1] + 3.0*pol[0]*h
        pol2 = pol[2] + 3.0*pol[0]*h*h + 2.0*pol[1]*h
        pol3 = pol[3] + pol[0]*h*h*h + pol[1]*h*h + pol[2]*h
        return(np.array([pol0, pol1, pol2, pol3]))
    if pol_ord == 2:
        pol0 = pol[0]
        pol1 = pol[1] + 2.0*pol[0]*h
        pol2 = pol[2] + pol[0]*h*h+pol[1]*h
        return(np.array([pol0, pol1, pol2]))
    if pol_ord == 1:
        pol0 = pol[0]
        pol1 = pol[1] + pol[0]*h
        return(np.array([pol0, pol1]))


def pol_d(pol, x):
    pol_ord = len(pol)-1
    if pol_ord == 3:
        return 3.0*pol[0]*x*x+2.0*pol[1]*x+pol[2]
    if pol_ord == 2:
        return 2.0*pol[0]*x+pol[1]
    if pol_ord == 1:
        return pol[0]  # *np.ones(len(np.array(x)))


def pol_dd(pol, x):
    pol_ord = len(pol)-1
    if pol_ord == 3:
        return 6.0*pol[0]*x+2.0*pol[1]
    if pol_ord == 2:
        return 2.0*pol[0]
    if pol_ord == 1:
        return 0.0


def pol_calc(pol, x):
    pol_f = np.poly1d(pol)
    return(pol_f(x))


xm_in_px = 3.675 / 85  # Lane width (12 ft in m) is ~85 px on image
ym_in_px = 3.048 / 24  # Dashed line length (10 ft in m) is ~24 px on image


def px_to_m(px):  # Conver ofset in pixels in x axis into m
    return xm_in_px*px

# Calculate offset from the lane center


def lane_offset(left, right):
    offset = IMAGE_W/2.0-(pol_calc(left, 1.0) + pol_calc(right, 1.0))/2.0
    return px_to_m(offset)


# Calculate radius of curvature of a line
MAX_RADIUS = 10000


def r_curv(pol, y):
    if len(pol) == 2:  # If the polinomial is a linear function
        return MAX_RADIUS
    else:
        y_pol = np.linspace(0, 1, num=EQUID_POINTS)
        x_pol = pol_calc(pol, y_pol)*xm_in_px
        y_pol = y_pol*IMAGE_H*ym_in_px
        pol = np.polyfit(y_pol, x_pol, len(pol)-1)
        d_y = pol_d(pol, y)
        dd_y = pol_dd(pol, y)
        r = ((np.sqrt(1+d_y**2))**3)/abs(dd_y)
        if r > MAX_RADIUS:
            r = MAX_RADIUS
        return r
# Calculate radius of curvature of a lane by avaraging lines curvatures


def lane_curv(left, right):
    l = r_curv(left, 1.0)
    r = r_curv(right, 1.0)
    if l < MAX_RADIUS and r < MAX_RADIUS:
        return (r_curv(left, 1.0)+r_curv(right, 1.0))/2.0
    else:
        if l < MAX_RADIUS:
            return l
        if r < MAX_RADIUS:
            return r
        return MAX_RADIUS


EQUID_POINTS = 25  # Number of points to use for the equidistant approximation


def equidistant(pol, d, max_l=1, plot=False):
    y_pol = np.linspace(0, max_l, num=EQUID_POINTS)
    x_pol = pol_calc(pol, y_pol)
    y_pol *= IMAGE_H  # Convert y coordinates to [0..223] scale
    x_m = []
    y_m = []
    k_m = []
    for i in range(len(x_pol)-1):
        # Calculate polints position between given points
        x_m.append((x_pol[i+1]-x_pol[i])/2.0+x_pol[i])
        y_m.append((y_pol[i+1]-y_pol[i])/2.0+y_pol[i])
        if x_pol[i+1] == x_pol[i]:
            k_m.append(1e8)  # A vary big number
        else:
            # Slope of perpendicular lines
            k_m.append(-(y_pol[i+1]-y_pol[i])/(x_pol[i+1]-x_pol[i]))
    x_m = np.array(x_m)
    y_m = np.array(y_m)
    k_m = np.array(k_m)
    # Calculate equidistant points
    y_eq = d*np.sqrt(1.0/(1+k_m**2))
    x_eq = np.zeros_like(y_eq)
    if d >= 0:
        for i in range(len(x_m)):
            if k_m[i] < 0:
                y_eq[i] = y_m[i]-abs(y_eq[i])
            else:
                y_eq[i] = y_m[i]+abs(y_eq[i])
            x_eq[i] = (x_m[i]-k_m[i]*y_m[i])+k_m[i]*y_eq[i]
    else:
        for i in range(len(x_m)):
            if k_m[i] < 0:
                y_eq[i] = y_m[i]+abs(y_eq[i])
            else:
                y_eq[i] = y_m[i]-abs(y_eq[i])
            x_eq[i] = (x_m[i]-k_m[i]*y_m[i])+k_m[i]*y_eq[i]
    y_eq /= IMAGE_H  # Convert all y coordinates back to [0..1] scale
    y_pol /= IMAGE_H
    y_m /= IMAGE_H
    # Fit equidistant with a polinomial
    pol_eq = np.polyfit(y_eq, x_eq, len(pol)-1)
    if plot:  # Visualize results
        plt.plot(x_pol, y_pol, color='red', linewidth=1,
                 label='Original line')  # Original line
        plt.plot(x_eq, y_eq, color='green', linewidth=1,
                 label='Equidistant')  # Equidistant
        plt.plot(pol_calc(pol_eq, y_pol), y_pol, color='blue',
                 linewidth=1, label='Approximation')  # Approximation
        plt.legend()
        for i in range(len(x_m)):
            plt.plot([x_m[i], x_eq[i]], [y_m[i], y_eq[i]],
                     color='black', linewidth=1)  # Draw connection lines
        plt.savefig('readme_img/equid.jpg')
    return pol_eq


# Choose the best polynomial order to fit points (x,y)
DEV_POL = 1.5  # Max mean squared error of the approximation
MSE_DEV = 1  # Minimum mean squared error ratio to consider higher order of the polynomial


def best_pol_ord(x, y):
    # print(x, y)
    pol1 = np.polyfit(y, x, 1)
    pred1 = pol_calc(pol1, y)
    mse1 = mean_squared_error(x, pred1)
    if mse1 < DEV_POL:
        # print(pol1, mse1, DEV_POL)
        return pol1, mse1
    pol2 = np.polyfit(y, x, 2)
    pred2 = pol_calc(pol2, y)
    mse2 = mean_squared_error(x, pred2)
    if mse2 < DEV_POL or mse1/mse2 < MSE_DEV:
        return pol2, mse2
    else:
        pol3 = np.polyfit(y, x, 3)
        pred3 = pol_calc(pol3, y)
        mse3 = mean_squared_error(x, pred3)
        if mse2/mse3 < MSE_DEV:
            return pol2, mse2
        else:
            return pol3, mse3


def smooth_dif_ord(pol_p, x, y, new_ord):
    x_p = pol_calc(pol_p, y)
    x_new = (x+x_p)/2.0
    return np.polyfit(y, x_new, new_ord)


def thres_r_calc(sens):
    thres = -0.0411*sens**2+9.1708*sens-430.0
    if sens < 210:
        if thres < sens/6:
            thres = sens/6
    else:
        if thres < 20:
            thres = 20
    return thres


def find(img, left=True, p_ord=POL_ORD, pol=np.zeros(POL_ORD+1), max_n=0):
    x_pos = []  # lists of found points
    y_pos = []
    max_l = img.shape[0]  # number of lines in the img
    for i in range(max_l-int(max_l*RANGE)):
        y = max_l-i  # Line number
        y_01 = y / float(max_l)  # y in [0..1] scale
        if abs(pol[-1]) > 0:  # If it not a still image or the first video frame
            if y_01 >= max_n + SPEED:  # If we can use pol to find center of the virtual sensor from the previous frame
                cent = int(pol_calc(pol, y_01-SPEED))
                if y == max_l:
                    if left:
                        cent = LEFT_LANE
                    else:
                        cent = RIGHT_LANE
            else:  # Prolong the pol tangentially
                k = pol_d(pol, max_n)
                b = pol_calc(pol, max_n)-k*max_n
                cent = int(k*y_01+b)
            if cent > IMAGE_W-WINDOW_SIZE:
                cent = IMAGE_W-WINDOW_SIZE
            if cent < WINDOW_SIZE:
                cent = WINDOW_SIZE
        else:  # If it is a still image
            if len(x_pos) > 0:  # If there are some points detected
                cent = x_pos[-1]  # Use the previous point as a senser center
            else:  # Initial guess on line position
                if left:
                    cent = LEFT_LANE
                else:
                    cent = RIGHT_LANE
        sens = img[max_l-1-i:max_l-i, cent-WINDOW_SIZE:cent +
                   WINDOW_SIZE, 2]  # Red channel only for right white line
        if len(sens[0, :]) < WINDOW_SIZE:  # If we out of the image
            break
        x_max = max(sens[0, :])  # Find maximal value on the sensor
        sens_mean = np.mean(sens[0, :])
        # Get threshold
        loc_thres = thres_r_calc(sens_mean)
        loc_dev = DEV
        if len(x_pos) == 0:
            loc_dev = WINDOW_SIZE
        if (x_max-sens_mean) > loc_thres and (x_max > 100 or left):
            if left:
                x = list(reversed(sens[0, :])).index(x_max)
                x = cent+WINDOW_SIZE-x
            else:
                x = list(sens[0, :]).index(x_max)
                x = cent-WINDOW_SIZE+x
            # if the sensor touchs black triangle
            if x-1 < LEFT_LANE*y_01 or x+1 > LEFT_LANE*y_01+RIGHT_LANE or\
                    np.count_nonzero(sens[0, :]) < WINDOW_SIZE:
                break  # We are done
            if abs(pol[-1]) < 1e-4:  # If there are no polynomial provided
                x_pos.append(x)
                y_pos.append(y_01)
            else:
                # If the found point deviated from expected position not significantly
                if abs(x-cent) < loc_dev:
                    x_pos.append(x)
                    y_pos.append(y_01)
    if len(x_pos) > 1:
        return x_pos, y_pos
    else:
        return [0.1], [0.1]

# drawlanes.py


def get_lane(img, plot=False):
    test = np.copy(img)
    warp = Frame(test)
    warp = warp.transform()
    # warp = transform(img, M)
    # img = undistort(img)
    ploty = np.linspace(0, 1, num=warp.shape[0])
    x2, y2 = find(warp)
    x, y = find(warp, False)
    right_fitx = pol_calc(best_pol_ord(x, y)[0], ploty)
    left_fitx = pol_calc(best_pol_ord(x2, y2)[0], ploty)
    y2 = np.int16(np.array(y2)*IMAGE_H)  # Convert into [0..223] scale
    y = np.int16(np.array(y)*IMAGE_H)
    return img,  left_fitx, right_fitx, ploty*IMAGE_H


def get_lane_video(img):
    global right_fit_p, left_fit_p, r_len, l_len, n_count, r_n, l_n
    sw = False
    test = np.copy(img)
    warp = Frame(test)
    warp = warp.transform()
    # img = undistort(img)
    if l_n < MAX_N and n_count > 0:
        x, y = find(warp, pol=left_fit_p, max_n=l_len)
    else:
        x, y = find(warp)
    if len(x) > MIN_POINTS:
        left_fit, mse_l = best_pol_ord(x, y)
        if mse_l > DEV_POL*9 and n_count > 0:
            left_fit = left_fit_p
            l_n += 1
        else:
            l_n /= 2
    else:
        left_fit = left_fit_p
        l_n += 1
    if r_n < MAX_N and n_count > 0:
        x2, y2 = find(warp, False, pol=right_fit_p, max_n=r_len)
    else:
        x2, y2 = find(warp, False)
    if len(x2) > MIN_POINTS:
        right_fit, mse_r = best_pol_ord(x2, y2)
        if mse_r > DEV_POL*9 and n_count > 0:
            right_fit = right_fit_p
            r_n += 1
        else:
            r_n /= 2
    else:
        right_fit = right_fit_p
        r_n += 1
    if n_count > 0:  # if not the first video frame
        # Apply filter
        if len(left_fit_p) == len(left_fit):  # If new and prev polinomial have the same order
            left_fit = pol_shift(left_fit_p, -SPEED)*(1.0-len(x) /
                                                      ((1.0-RANGE)*IMAGE_H))+left_fit*(len(x)/((1.0-RANGE)*IMAGE_H))
        else:
            left_fit = smooth_dif_ord(left_fit_p, x, y, len(left_fit)-1)
        l_len = y[-1]
        if len(right_fit_p) == len(right_fit):
            right_fit = pol_shift(right_fit_p, -SPEED)*(1.0-len(x2) /
                                                        ((1.0-RANGE)*IMAGE_H))+right_fit*(len(x2)/((1.0-RANGE)*IMAGE_H))
        else:
            right_fit = smooth_dif_ord(right_fit_p, x2, y2, len(right_fit)-1)
        r_len = y2[-1]

    if len(x) > MIN_POINTS and len(x2) <= MIN_POINTS:  # If we have only left line
        lane_w = pol_calc(right_fit_p, 1.0)-pol_calc(left_fit_p, 1.0)
        right_fit = smooth_dif_ord(right_fit_p, pol_calc(equidistant(left_fit, lane_w, max_l=l_len), y),
                                   y, len(left_fit)-1)
        r_len = l_len
        r_n /= 2
    if len(x2) > MIN_POINTS and len(x) <= MIN_POINTS:  # If we have only right line
        lane_w = pol_calc(right_fit_p, 1.0)-pol_calc(left_fit_p, 1.0)
        # print(lane_w)
        left_fit = smooth_dif_ord(left_fit_p, pol_calc(equidistant(right_fit, -lane_w, max_l=r_len), y2),
                                  y2, len(right_fit)-1)
        l_len = r_len
        l_n /= 2
    if (l_n < MAX_N and r_n < MAX_N):
        max_y = max(RANGE, l_len, r_len)
    else:
        max_y = 1.0  # max(RANGE, l_len, r_len)
        sw = True
    d1 = pol_calc(right_fit, 1.0)-pol_calc(left_fit, 1.0)
    dm = pol_calc(right_fit, max_y)-pol_calc(left_fit, max_y)
    if (d1 > MAX or d1 < 60 or dm < 0):
        left_fit = left_fit_p
        right_fit = right_fit_p
        l_n += 1
        r_n += 1
    ploty = np.linspace(max_y, 1, num=IMAGE_H)
    left_fitx = pol_calc(left_fit, ploty)
    right_fitx = pol_calc(right_fit, ploty)
    right_fit_p = np.copy(right_fit)
    left_fit_p = np.copy(left_fit)
    n_count += 1
    return img,  left_fitx, right_fitx, ploty*IMAGE_H, left_fit, right_fit


def draw_lane(img, video=True):  # Draw found lane line onto a normal image
    init_params(0)
    if video:
        img, left_fitx, right_fitx, ploty, left, right = get_lane_video(img)
    else:
        img, left_fitx, right_fitx, ploty = get_lane(img, True)
    warp = Frame(img)
    warp = warp.transform()
    warp_zero = np.zeros((IMAGE_H, IMAGE_W)).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    # result = cv2.addWeighted(img, 1.0, newwarp, 0.6, 0)
    result = cv2.addWeighted(warp, 1.0, color_warp, 0.6, 0)
    # result = newwarp
    if video:
        # # Add text information on the video frame
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # text_pos = 'Pos of the car: ' + \
        #     str(np.round(lane_offset(left, right), 2)) + ' m'
        # radius = np.round(lane_curv(left, right), 2)
        # if radius >= MAX_RADIUS:
        #     radius = 'Inf'
        # else:
        #     radius = str(radius)
        # text_rad = 'Radius: '+radius + ' m'
        # cv2.putText(result, text_pos, (10, 25), font, 1, (255, 255, 255), 2)
        # cv2.putText(result, text_rad, (10, 75), font, 1, (255, 255, 255), 2)
        return(result)
    else:
        return(result)


def init_params(ran):
    global right_fit_p, left_fit_p, n_count, RANGE, MIN_POINTS
    right_fit_p = np.zeros(POL_ORD+1)
    left_fit_p = np.zeros(POL_ORD+1)
    n_count = 0
    RANGE = ran
    MIN_POINTS = 25-15*ran
