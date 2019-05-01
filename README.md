# jalopy
Self-driving system for Euro Truck Simulator 2

## Can I run it?
* _(note)_ Windows comes packaged with something called a [Desktop Duplication API](https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/desktop-dup-api) that uses fast DXGI and Direct3D libraries to produce fast lighning-fast screenshots. I use a package called ```D3DShot``` for this very purpose. It can take 800x600 screenshots at around 50 FPS. ```MSS``` is also pretty good, and is a great generic solution for MacOS and Linux, but beware, I haven't yet tested those platforms for complete compatibility with Jalopy.
* Windows >8.1 (I'll figure out how to send key inputs to Unix operating systems later)
* Python >3.6 (64-bit installation only)

## Installation (tested on Windows 10)
* _(optional)_ Install [Anaconda](https://anaconda.com) for ease of installation
* _(conditional)_ If you use Visual Studio Code, add ```    "python.linting.pylintArgs": ["--extension-pkg-whitelist=cv2"]
``` to your ```settings.json``` file
1. Make sure you have pip installed by running ```pip --version``` on your computer. If not, navigate to the _jalopy_ folder and run ```python get-pip.py```
1. Install necessary packages (OpenCV, sklearn, etc.) with ```pip install -r packages.txt```

## How to run Jalopy
* _(optional)_ If you are running Jalopy on laptop, you probably want to plug in your computer to prevent battery consumption conservation from slowing down the game/OpenCV/Jalopy
<!-- 1. Run Euro Truck Simulator 2 at 800x600 resolution by changing your ```config.cfg``` file by modifying this lines to the specified values:
~~~
uset r_fullscreen "0"
uset r_mode_height "600"
uset r_mode_width "800"
~~~

Ideally, Jalopy could do this for you  -->
1. Place Euro Truck Simulator 2 to the upper left corner of your screen
1. Navigate to the main directory with ```cd jalopy```
1. Run the main .py file with ```python main.py```

## Update history
* TP1 : Quick demo of lane detection, working on streaming the game output to the OpenCV feed. Using personal gameplay and YouTube videos for testing purposes, transcoding on Handshake for optimal filesize and resolution.
* TP2 : Working self-driving demo with lane-slope heuristic. However, this heuristic is awful and should be discarded in favor of a cNN (coming TP3)

## Notes
* PIL's ImageGrab functionality is way, way, way too slow for _jalopy_. That package took around half a second to process a single screenshot, and that's *before* I even use OpenCV. Ew!

## Helpful links for learning
* [MIT OpenCourseWare](https://www.youtube.com/watch?v=1L0TKZQcUtA)
* [Hough lines](https://en.wikipedia.org/wiki/Hough_transform)
* [Hough lines implemented in OpenCV](https://docs.opencv.org/2.4.13.7/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html)
* [NumPy arrays vs. Python lists](https://stackoverflow.com/questions/993984/what-are-the-advantages-of-numpy-over-regular-python-lists)
