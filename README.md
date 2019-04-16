# jalopy
Self-driving system for Euro Truck Simulator 2

## Installation (tested on Windows 10)
* _(optional)_ Install [Anaconda](https://anaconda.com) for ease of installation
* _(conditional)_ If you use Visual Studio Code, add ```    "python.linting.pylintArgs": ["--extension-pkg-whitelist=cv2"]
``` to your ```settings.json``` file
1. Install OpenCV with ```pip install opencv-contrib-python```
1. Install matplotlib with ```pip install matplotlib```

## Update history
* TP1 : Quick demo of lane detection, working on streaming the game output to the OpenCV feed. Using personal gameplay and YouTube videos for testing purposes, transcoding on Handshake for optimal filesize and resolution.

## Helpful links for learning
* [MIT OpenCourseWare](https://www.youtube.com/watch?v=1L0TKZQcUtA)
* [Hough lines](https://en.wikipedia.org/wiki/Hough_transform)
* [Hough lines implemented in OpenCV](https://docs.opencv.org/2.4.13.7/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html)
* [NumPy arrays vs. Python lists](https://stackoverflow.com/questions/993984/what-are-the-advantages-of-numpy-over-regular-python-lists)