# Looking Glass Bridge SDK · Python Edition 

Integrate your existing Python-based 3D workflow with any Looking Glass display and treat the holographic panel as a second, light-field monitor. This package wraps the native Bridge SDK so you can render quilts in OpenGL, hand them off to Bridge, and see them instantly on the device – all without leaving Python.

## Features

* **Simple setup –** use the included Looking Glass Bridge driver or download the latest version from the Looking Glass Bridge website. 
* **Cross-platform wheels –** pre-built for Windows (x86-64), macOS (universal2), and manylinux (x86-64/arm64); each wheel already bundles the correct Bridge driver so you can `pip install` and go. 
* **Reference camera & quilt math** – helper classes to distort projection matrices and assemble quilts.  
* **Examples included** – `RotatingCube.py` and `SolarSystem.py` show minimal and advanced pipelines.  
* **Zero-copy texture path** – pass existing OpenGL textures; no CPU read-backs required.  

## Installation

Setup a virtual environment:

```bash
# Create virtual environment in .venv (Linux)
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

```bash
# Create virtual environment in .venv (macOS)
python -m venv .venv

# Activate it
source .venv/bin/activate
```

```bash
# Create virtual environment in venv (Windows)
python -m venv venv

# Activate it
"venv\Scripts\activate"
```

```bash
pip install "bridge-python-sdk @ git+https://github.com/Looking-Glass/bridge-python-sdk"
```

For a quick burn in test, run one of the examples:

```bash
python -m bridge_python_sdk.Examples.SolarSystem
python -m bridge_python_sdk.Examples.RotatingCube
python -m bridge_python_sdk.Examples.DisplayQuilt /path/to/quilt_qs8x6a0.75.png
python -m bridge_python_sdk.Examples.DisplayRGBD /path/to/rgbd_image.png
```

### Note: you will need to run the activate command again

**Known issues:**

DisplayQuilt and DisplayRGBD samples both support video, but currently not at full speed.

On Linux you must run under **X11**; Wayland is not yet supported.

## Quick Start

See `src/bridge_python_sdk/Examples` for fully-worked programmes that spin a cube and orbit a solar system.

## Tutorials

We have a few tutorials in the 'docs' folder, namely:

* [Getting Started](./docs/GettingStarted.md) | Covers setting up a python environment and installing the sdk
* [Displaying a Prerendered Quilt](./docs/DisplayingQuilt.md) | Covers reading a quilt image from a file and displaying it
* [Displaying a Prerendered RGBD](./docs/DisplayingRGBD.md) | Covers reading an RGBD image from a file and displaying it
* [Rendering a Quilt](./docs/RenderingAQuilt.md) | Covers rendering a quilt in real time and displaying it

## Prerequisites

Your application (or the provided examples) must be able to:

1. **Render multiple views** of the scene per frame.  
2. **Distort the projection matrix** supplied by the SDK.  
3. **Render into an OpenGL texture** that the SDK can access.  

## Documentation

* **Camera model** – learn how eye positions are generated for each view.  
* **Quilts** – understand the tiled texture fed to Bridge.  
* **How Looking Glass works** – deep dive into light-field rendering.

All docs live at <https://docs.lookingglassfactory.com>.

## Support

* **Email:** <support@lookingglassfactory.com>  
* **Discord:** <https://discord.gg/2GvnNMQhxF>  
* **Issues:** <https://github.com/Looking-Glass/bridge-python-sdk/issues>

## License

This SDK is released under the MIT License. See `LICENSE` for details.