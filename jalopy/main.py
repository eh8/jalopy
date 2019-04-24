# Eric Cheng

# Imaging libraries (mss is faster than standard PIL)
from mss import mss
from PIL import Image

# Controlling game state
from keyPressed import ReleaseKey, PressKey, W, A, S, D

# Actual OpenCV
import numpy as np
import cv2

# Debugging
import time
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter()


# def process_image(original_image):
#     processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
#     processed_image = cv2.Canny(original_image, threshold1=200, threshold2=300)
#     return processed_image


def countDown():
    for i in list(range(7))[::-1]:
        print(i+1)
        time.sleep(1)


def boot():
    while True:
        roi = {'top': 35, 'left': 0, 'width': 800, 'height': 600}
        screenshot = mss()

        screenshot.get_pixels(roi)
        image = Image.frombytes('RGB', (screenshot.width, screenshot.height),
                                screenshot.image)
        screen = np.array(image)
        new_screen = process_image(screen)

        print('Down')
        PressKey(W)
        time.sleep(3)
        print('Up')
        PressKey(W)

        cv2.imshow('Jalopy', cv2.cvtColor(new_screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def main():
    countDown()
    boot()


if __name__ == '__main__':
    main()
