# Requirements
Libraries
* opencv (included with opencv-python)
* ffmpeg (included with opencv-python)

Python
* opencv-python
* tensorflow
* playsound

# Windows
Using miniconda is a convenient way to get everything working:
https://docs.conda.io/en/latest/miniconda.html

# OSX M1
There aren't pre-built packages on pip for M1 processors it seems.
https://www.tensorflow.org/install/source#macos

If building from source doesn't work, try
```
pip install tensorflow-macos
```

```
brew install gst-plugins-base gst-plugins-good gst-plugins-ugly gst-plugins-bad
```

```
brew tap homebrew/cask-drivers
brew install --cask cameracontroller
```
