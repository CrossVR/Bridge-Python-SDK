#!/usr/bin/env python3
# MinimalQuilt.py
import sys
import os
import io
import urllib.request
import glfw

from PIL import Image
import numpy as np
from OpenGL import GL

# Add the parent directory to the Python path 
# This is only needed to make the example work if called from the installed pip package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BridgeApi import BridgeAPI, PixelFormats

# Download an example quilt from blocks
with urllib.request.urlopen("https://s3.amazonaws.com/lkg-blocks/legacy/781/source.png") as resp:
    data_bytes = resp.read()

# Convert the png bytes into a Pillow image
image = Image.open(io.BytesIO(data_bytes)).convert("RGBA")

# Get the raw RGBA data from Pillow iamge
data = np.array(image, dtype=np.uint8)

# define required data about the quilt.
h, w, _ = data.shape
cols = 5
rows = 9 
aspect = 0.75
frame_size = w * h * 4

# Initialize glfw and create a hidden dummy window for a GL context
if not glfw.init():
    print("Error: failed to initialize GLFW", file=sys.stderr)
    sys.exit(1)

# Set the glfw window to be hidden, have a size of 1 x 1 and create it
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
dummy = glfw.create_window(1, 1, "", None, None)

# Exit if the glfw window failed for any reason
if not dummy:
    print("Error: failed to create hidden GLFW window", file=sys.stderr)
    glfw.terminate()
    sys.exit(1)

# Make sure the current thread owns the opengl context.
glfw.make_context_current(dummy)

# Create the bridge API object
bridge = BridgeAPI()

# Initialize the API with some descriptive name 
if not bridge.initialize("DisplayQuilt"):
    # if the bridge api fails for some reason exit
    print("Bridge initialize failed", file=sys.stderr)
    glfw.destroy_window(dummy)
    glfw.terminate()
    sys.exit(1)

# Instance a bridge window
# The input parameter here allows you to support multiple connected looking glass displays on one computer.
# If you are using only one display you can use -1 which will default to the first looking glass display connected.
br_wnd = bridge.instance_window_gl(-1)
if br_wnd == 0:
    print("Bridge.instance_window_gl returned 0", file=sys.stderr)
    glfw.destroy_window(dummy)
    glfw.terminate()
    sys.exit(1)

# Create GL texture to store the quilt
tex = GL.glGenTextures(1)
GL.glBindTexture(GL.GL_TEXTURE_2D, tex)

# Linear texture filtering looks the best in our testing
GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
GL.glTexImage2D(
    GL.GL_TEXTURE_2D,
    0,
    GL.GL_RGBA8,
    w,
    h,
    0,
    GL.GL_RGBA,
    GL.GL_UNSIGNED_BYTE,
    data,
)

# Main loop on the hidden window
while not glfw.window_should_close(dummy):
    # This is the call into bridge that actually renders the quilt texture onto the display.
    bridge.draw_interop_quilt_texture_gl(
        br_wnd,
        tex,
        PixelFormats.RGBA,
        w,
        h,
        cols,
        rows,
        aspect,
        1.0, # This is the zoom parameter, 1.0 is the default zoom and will fit the quilt to the screen
    )
    glfw.swap_buffers(dummy)
    glfw.poll_events()

# when the window exits dont forget to clean up!
GL.glDeleteTextures(1, [tex])
glfw.destroy_window(dummy)
glfw.terminate()