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
