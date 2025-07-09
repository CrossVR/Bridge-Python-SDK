# Setting Up To Render | Displaying a Prerendered Quilt

This tutorial is intended to follow directly after the [Getting Started](./GettingStarted.md) tutorial, if you did not follow those steps you will need to setup a virtual python environment and install the python bridge sdk yourself.

## Overview

A looking glass display projects many different views of a scene into the space in front of it, allowing you to move and look around objects shown in the display. This means we need to deliver a set of views to the display to be shown. The main way that we achieve this is using a grid of views stored in a single image which we call a [quilt](https://docs.lookingglassfactory.com/keyconcepts/quilts).

Before we worry about how to render a quilt we first need to setup the bridge integration, the high level overview is as follows:

1. Get a Quilt
2. Initialize OpenGL
3. Initialize Bridge
4. Setup Quilt Texture
5. Send Bridge the Quilt

For now we will just read a quilt texture from a url, but in the next tutorial in the series we will render the quilt in the same code.

## Get a Quilt

First lets download a quilt and convert it into raw data to upload to an opengl texture.

```python
# Download an example quilt from blocks
with urllib.request.urlopen("https://s3.amazonaws.com/lkg-blocks/legacy/781/source.png") as resp:
    data_bytes = resp.read()

# Convert the png bytes into a Pillow image
image = Image.open(io.BytesIO(data_bytes)).convert("RGBA")

# Get the raw RGBA data from Pillow image
data = np.array(image, dtype=np.uint8)

# define required data about the quilt.
h, w, _ = data.shape
cols = 5
rows = 9 
aspect = 0.75
frame_size = w * h * 4
```

## Initialize OpenGL

Now that we have a quilt and all the associated data, we can move on to the opengl stuff.

The bridge pip package already installs PyOpenGL and glfw for opengl support, Pillow for image support, and numpy to help with data structures, so we should have everything we need to get started.

Below we initialize glfw, and create a hidden window as we do not need any interaction or 

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

```python
# Create the bridge API object
bridge = BridgeAPI()

# Initialize the API with some descriptive name 
if not bridge.initialize("DisplayQuilt"):
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
```

## Setup Quilt Texture

We are finally almost done, we now just need to create an opengl texture from the quilt image data we generated in step 1.

```python
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
```

## Send Bridge the Quilt

Ok, we are finally here, the final step. 

Just to review, we have downloaded a quilt from blocks and converted it into raw RGBA data. Then we initialized opengl, keeping the 2D window hidden. We initialized bridge and instanced a window on the first looking glass display connected. Finally we created a texture on the GPU to store the quilt image data.

All we have left to do is send that quilt texture to bridge and wait for the user to close the application.

```python
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
```

You should have a python file that looks something like [this.](../src/bridge_python_sdk/Examples/MinimalQuilt.py)

### Next tutorial