# Setting Up To Render | Displaying a Prerendered RGBD

This tutorial is intended to follow directly after the [Getting Started](./GettingStarted.md) tutorial, if you did not follow those steps you will need to setup a virtual python environment and install the python bridge sdk yourself.

This is VERY similar to the [quilt tutorial](./DisplayingQuilt.md) but uses a slightly different api call from bridge.

## Overview

A looking glass display projects many different views of a scene into the space in front of it, allowing you to move and look around objects shown in the display. This means we need to deliver a set of views to the display to be shown. The main way that we achieve this is using a grid of views stored in a single image which we call a [quilt](https://docs.lookingglassfactory.com/keyconcepts/quilts).

What if you cannot make a quilt?

Say for instance you will only have a single 2D view of a scene, like a photo from a cell phone. You obviously could move the cellphone around and take many photos but this is sometimes impossible, maybe you took the photo 20 years ago. What can you do in this case?

To help in this situation we support a format we call RGBD, which is a single image with color on one side and a greyscale depth map on the other side. It is very easy to generate these images using AI depth generation models such as [Distill Any Depth](https://docs.lookingglassfactory.com/community/convert-any-image-into-a-hologram#convert-with-other-ai-tools-distill-any-depth)

As with the quilt tutorial, the high level overview is as follows:

1. Get an RGBD
2. Initialize OpenGL
3. Initialize Bridge
4. Setup RGBD Texture
5. Send Bridge the RGBD

For now we will just read a RGBD texture from a url, but in a future tutorial in this series we will show how to generate the depth map for any 2d image.

## Get an RGBD

First lets download an RGBD and convert it into raw data to upload to an opengl texture.

```python
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
```

## Initialize OpenGL

Now that we have an RGBD and all the associated data, we can move on to the opengl stuff.

The bridge pip package already installs PyOpenGL and glfw for opengl support, Pillow for image support, and numpy to help with data structures, so we should have everything we need to get started.

Below we initialize glfw, and create a hidden window as we will not render anything to the 2D window.

```python
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
```

## Initialize Bridge

After opengl is initialized we can bring up the bridge api integration. The Python Bridge SDK includes the necessary runtime files, but if you have a newer version of bridge installed it will use that instead. 

We always recommend installing the latest version of bridge from [here](https://lookingglassfactory.com/software/looking-glass-bridge).

### NOTE: This is slightly different from the quilt tutorial

Because we do not have a quilt, we have an RGBD, we are going to have the quilt rendered by bridge. But to retain as much flexibility the bridge api requires you to specify the quilt settings you want to render with. This means we need to do the extra step of obtaioning the default quilt settings for the currently attached display as show at the end of the next codeblock.

RGBD rendering also have two extra parameters, focus and depthiness. I show how to normalize this so that the focus values correspond with the depth map colors.

```python
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
```

## Setup RGBD Texture

We are finally almost done, we now just need to create an opengl texture from the RGBD image data we generated in step 1.

```python
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
```

## Send Bridge the RGBD

Ok, we are finally here, the final step. 

Just to review, we have downloaded an RGBD from blocks and converted it into raw image data. Then we initialized opengl, keeping the 2D window hidden. We initialized bridge and instanced a window on the first looking glass display connected. We then queried bridge for the default quilt settings for this display, and then created a texture on the GPU to store the RGBD image data.

All we have left to do is send that RGBD texture to bridge and wait for the user to close the application.

```python
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
```

You should have a python file that looks something like [this.](../src/bridge_python_sdk/Examples/MinimalRGBD.py)

### Next tutorial