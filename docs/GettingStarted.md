# Getting Started

The Python Bridge SDK allows you to focus on the parts that matter, the rendering of your 3D scene, while Looking Glass Bridge does the heavy lifting to display your rendered scene on a looking glass display.

This document will focus on installing and testing the Python Bridge SDK, then we will move on to displaying a prerendered quilt on the looking glass. From there we will then describe how to render a quilt and display it in realtime, allowing for interactive applications.

## Creating your environment

Step zero is to install Python 3, but I will leave that as an exercise for the reader. The documentation from [python.org](https://wiki.python.org/moin/BeginnersGuide/Download) should suffice.

The first step (after installing python) to getting started with the Python Bridge SDK is to create a python virtual environment, I will be using venv but conda or any other python virtual environment should work.

If you skip this step you can run into issues where different python projects require different library versions and those conflicts are very hard to deal with without a virtual environment.

```bash
mkdir project_directory
cd project_directory
python -m venv venv
```

This will create a folder in your project directory called `venv` which holds the python virtual environment that we will be using.

Whenever you are using a python virtual environment you will need to activate it every time you open a new terminal window, to do this simply:

```bash
cd project_directory
"./venv/Scripts/activate"
```

This is the command to activate a venv on windows, other os's will look slightly different. If you are on macos or linux, [the readme](../README.md) has the instructions for all operating systems.

## Installing the Python Bridge SDK

Now we need to install the package for the Python Bridge SDK. This can be done via pip.

```bash
pip install "bridge-python-sdk @ git+https://github.com/Looking-Glass/bridge-python-sdk"
```

This will install the latest version of the Python Bridge SDK from the git repo. 

## Test

To run a simple test of the integration you can run this command:

```bash
python -m bridge_python_sdk.Examples.RotatingCube
```

This will open a window with a rotating cube, displayed in a 2D window on your desktop and, provided you have a looking glass connected, on your looking glass display.

You can view the code for this example here: [RotatingCube.py](../src/bridge_python_sdk/Examples/RotatingCube.py). This uses the very basic render engine implemented [here](../src/bridge_python_sdk/Rendering/).

We will create a single file rendering example in the next tutorial.

### Next tutorial

The next tutorial is split into two almost duplicate sections. One for displaying a quilt using bridge, and one for displaying an RGBD using bridge.

You can find them here: [Displaying a Prerendered Quilt](./DisplayingQuilt.md) and here: [Displaying a Prerendered RGBD](./DisplayingRGBD.md)