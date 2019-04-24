import numpy as np
import cv2
from mss import mss
from PIL import Image

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