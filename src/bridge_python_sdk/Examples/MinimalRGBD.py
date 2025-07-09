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

# Download an example RGBD from blocks
with urllib.request.urlopen("https://s3.amazonaws.com/lkg-blocks/u/72d8084888a8489c/rgbd.png") as resp:
    data_bytes = resp.read()

# Convert the png bytes into a Pillow image
image = Image.open(io.BytesIO(data_bytes)).convert("RGBA")

# Get the raw RGBA data from Pillow image
data = np.array(image, dtype=np.uint8)

# Define height and width parameters based on the input the image
h, w, _ = data.shape

# This is determined by which side the depth is stored in the image. 
# 0 = top, 1 = bottom, 2 = right, 3 = left (default: 2).
depth_loc = 2 

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
if not bridge.initialize("DisplayRGBD"):
    # If the bridge api fails for some reason exit
    print("Bridge initialize failed", file=sys.stderr)
    glfw.destroy_window(dummy)
    glfw.terminate()
    sys.exit(1)

# Instance a bridge window
# The input parameter here allows you to support multiple connected looking glass displays on one computer.
# If you are using only one display you can use -1 which will default to the first looking glass display connected.
br_wnd = bridge.instance_window_gl(-1)
if br_wnd == 0:
    print("Bridge.instance_window_gl failed", file=sys.stderr)
    glfw.destroy_window(dummy)
    glfw.terminate()
    sys.exit(1)

# Now that we have a window we can query the displays default quilt settings
asp, quiltWidth, quiltHeight, cols, rows = bridge.get_default_quilt_settings(br_wnd)
aspect = float(asp)

# -1 focuses on the farthest part of the image (black in the depth map)
#  0 focuses on the center
#  1 focuses on the closest part of the image (white in the depth map)
focus_input = 0 

# 0 renders the RGBD completely flat
# 1 is a good default, but you can go higher if needed.
depthiness_input = 1 

# Normalize focus.
focus_min = 0.005
focus_max = -0.007
normalized_focus = focus_min + ((((focus_input * depthiness_input)) + 1.0) / 2.0) * (focus_max - focus_min)

# Create GL texture to store the RGBD
tex = GL.glGenTextures(1)
GL.glBindTexture(GL.GL_TEXTURE_2D, tex)

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
    # This is the call into bridge that actually renders the RGBD texture onto the display.
    # Bridge will render the RGBD to an internal quilt with the specified quilt settings and then display that quilt on screen.
    bridge.draw_interop_rgbd_texture_gl(
        br_wnd,
        tex,
        PixelFormats.RGBA,
        w,               # input width
        h,               # input height
        quiltWidth,      # quilt width
        quiltHeight,     # quilt height
        cols,            # quilt columns
        rows,            # quilt rows
        aspect,          # display aspect
        normalized_focus,# normalized focus
        depthiness_input,# depthiness
        1.0,             # This is the zoom parameter, 1.0 is the default zoom and will fit the quilt to the screen
        depth_loc        # depth location
    )
    glfw.swap_buffers(dummy)
    glfw.poll_events()

# when the window exits dont forget to clean up!
GL.glDeleteTextures(1, [tex])
glfw.destroy_window(dummy)
glfw.terminate()