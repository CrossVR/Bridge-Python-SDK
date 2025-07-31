import math
import time
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ctypes
import numpy as np
from OpenGL import GL
import glfw
from typing import Optional, Dict, Any, List, Tuple

from BridgeApi import BridgeAPI, PixelFormats

class Mesh:
    # This is storage for the raw mesh data for a cube. 
    # There are many ways to store mesh data and this is the most simple,
    # 3 floats per triangle, 2 triangles per face, 6 faces
    cube_vertices = np.array([
        # Front face
        -0.5,-0.5, 0.5,  0.5,-0.5, 0.5,  0.5, 0.5, 0.5,
         0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5,-0.5, 0.5,
        # Back face
        -0.5,-0.5,-0.5, -0.5, 0.5,-0.5,  0.5, 0.5,-0.5,
         0.5, 0.5,-0.5,  0.5,-0.5,-0.5, -0.5,-0.5,-0.5,
        # Left face
        -0.5, 0.5,-0.5, -0.5, 0.5, 0.5, -0.5,-0.5, 0.5,
        -0.5,-0.5, 0.5, -0.5,-0.5,-0.5, -0.5, 0.5,-0.5,
        # Right face
         0.5, 0.5, 0.5,  0.5, 0.5,-0.5,  0.5,-0.5,-0.5,
         0.5,-0.5,-0.5,  0.5,-0.5, 0.5,  0.5, 0.5, 0.5,
        # Top face
        -0.5, 0.5,-0.5,  0.5, 0.5,-0.5,  0.5, 0.5, 0.5,
         0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5,-0.5,
        # Bottom face
        -0.5,-0.5,-0.5, -0.5,-0.5, 0.5,  0.5,-0.5, 0.5,
         0.5,-0.5, 0.5,  0.5,-0.5,-0.5, -0.5,-0.5,-0.5
    ], dtype=np.float32)

    # This is information about how the triangles stored in the mesh
    # data above are laid out in memory, because opengl supports many 
    # ways to store mesh data in memory we need to specify exactly how
    # our mesh is stored.
    cube_stride = 3 * 4
    cube_attribs = [(0, 3, GL.GL_FLOAT, False, cube_stride, 0)]

    # This is a simple helper function that we use to 
    # rotate the cube so the demo is less boring
    @staticmethod
    def euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)

        rx_m = np.array([[1,0,0,0],
                         [0,cx,-sx,0],
                         [0,sx,cx,0],
                         [0,0,0,1]], dtype=np.float32)
        ry_m = np.array([[cy,0,sy,0],
                         [0,1,0,0],
                         [-sy,0,cy,0],
                         [0,0,0,1]], dtype=np.float32)
        rz_m = np.array([[cz,-sz,0,0],
                         [sz,cz,0,0],
                         [0,0,1,0],
                         [0,0,0,1]], dtype=np.float32)
        return rz_m @ ry_m @ rx_m

    def __init__(self,
                 vertices: Optional[np.ndarray] = None,
                 attribs: Optional[List[tuple]] = None):
        vertices = Mesh.cube_vertices if vertices is None else vertices
        attribs  = Mesh.cube_attribs  if attribs  is None else attribs
 
        # Convert the vertices data into the correct raw data for opengl
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self._vertices_ctypes = (ctypes.c_float * vertices.size).from_buffer(vertices)
        self.vertex_count = vertices.nbytes // attribs[0][4]

        # Create the backing data storage for the vertex attribute data
        # This data describes to opengl how your vertex data is laid out in memory
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # Create the backing data storage for the vertex array
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes,
                        self._vertices_ctypes, GL.GL_STATIC_DRAW)

        for loc, size, typ, norm, stride, offset in attribs:
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, size, typ, norm, stride,
                                     ctypes.c_void_p(offset))

        # Bind the vbo to the vao
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

    def draw(self, mode: int = GL.GL_TRIANGLES) -> None:
        # To draw we essentially activate the mesh by binding it
        GL.glBindVertexArray(self.vao)
        # Then ask opengl to draw the array
        GL.glDrawArrays(mode, 0, self.vertex_count)
        # Bind to 0 (null) to deactivate your mesh
        GL.glBindVertexArray(0)

    # Don't forget to cleanup after yourself
    def __del__(self):
        try:
            GL.glDeleteBuffers(1, [self.vbo])
            GL.glDeleteVertexArrays(1, [self.vao])
        except Exception:
            pass

class Shader:
    # This is a very basic vertex shader
    # It uses the u_mvp matrix to transform the vertices into the correct 
    # screen space location, and calculates a uv coordinate for the vertex
    DEFAULT_VERTEX_SRC = """
    #version 330 core
    layout(location = 0) in vec3 aPos;
    uniform mat4 u_mvp;
    out vec3 vPos;
    void main()
    {
        gl_Position = u_mvp * vec4(aPos, 1.0);
        vPos = aPos;
    }
    """

    # This is a very basic fragment shader
    # It uses the uv coordinate calculated from the vertex shader (vUV) to 
    # color the cube something
    DEFAULT_FRAGMENT_SRC = """
    #version 330 core
    in vec3 vPos;
    out vec4 FragColor;
    void main()
    {
        vec3 N = normalize(cross(dFdx(vPos), dFdy(vPos)));
        vec3 aN = abs(N);
        vec2 uv;
        if (aN.z >= aN.x && aN.z >= aN.y)
            uv = vPos.xy + 0.5;
        else if (aN.x >= aN.y)
            uv = vec2(vPos.z, vPos.y) + 0.5;
        else
            uv = vec2(vPos.x, vPos.z) + 0.5;

        float range = 0.02;
        vec3 col = vec3(uv, 1.0 - uv.x);
        if (uv.x < range || uv.x > 1.0 - range ||
            uv.y < range || uv.y > 1.0 - range)
            col = vec3(0.0);

        FragColor = vec4(col, 1.0);
    }
    """

    def __init__(self,
                 vertex_src: Optional[str] = None,
                 fragment_src: Optional[str] = None):
        vertex_src = vertex_src or Shader.DEFAULT_VERTEX_SRC
        fragment_src = fragment_src or Shader.DEFAULT_FRAGMENT_SRC

        # Create storage on the GPU for the shader program
        self.id = GL.glCreateProgram()

        # Compile the vertex and fragment shader code into actual programs
        vert = self._compile(vertex_src, GL.GL_VERTEX_SHADER)
        frag = self._compile(fragment_src, GL.GL_FRAGMENT_SHADER)

        # Send the compiled shader programs to the gpu
        GL.glAttachShader(self.id, vert)
        GL.glAttachShader(self.id, frag)

        # Link the vertex and fragment shaders together and make sure there is no error
        GL.glLinkProgram(self.id)
        if not GL.glGetProgramiv(self.id, GL.GL_LINK_STATUS):
            log = GL.glGetProgramInfoLog(self.id).decode()
            GL.glDeleteProgram(self.id)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            raise RuntimeError(f"Program link failed:\n{log}")

        # Cleanup after yourself!
        GL.glDeleteShader(vert)
        GL.glDeleteShader(frag)

    # This is how the shader compilation is done
    # We do a bit of extra work to make the error messages easier
    # to understand.
    def _compile(self, src: str, shader_type):
        # Create the shader
        shader = GL.glCreateShader(shader_type)

        # Load and compile the source code
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)

        # Check for errors and print human readable error if needed 
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(shader).decode()
            GL.glDeleteShader(shader)
            type_name = {GL.GL_VERTEX_SHADER: "vertex", GL.GL_FRAGMENT_SHADER: "fragment"}.get(shader_type, str(shader_type))
            numbered_src = '\n'.join(f"{i + 1:4d}: {line}" for i, line in enumerate(src.splitlines()))
            raise RuntimeError(f"{type_name.capitalize()} shader compile failed:\n"
                               f"{log}\nSource with line numbers:\n{numbered_src}")
        return shader
    
    # This is how we activate this shader, we will use it in the next class 
    def use(self):
        GL.glUseProgram(self.id)

class Render:
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        title: str = "",
        fov: float = 14.0,
        near: float = 0.5,
        far: float = 1000.0,
        camera_distance: float = 10.0,
        lkg_size: float = 3.0,
        lkg_center: Optional[np.ndarray] = None,
        lkg_up: Optional[np.ndarray] = None,
        lkg_viewcone: float = 40.0
    ) -> None:
        # Initialize opengl
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)

        self._width = width
        self._height = height

        # Create a window
        self._window = glfw.create_window(width, height, title, None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)

        GL.glEnable(GL.GL_DEPTH_TEST)

        # We use the LKGCamera for both the 2D window and the Quilt render
        # so we need to set it up even if bridge fails to initialize
        fbw, fbh = glfw.get_framebuffer_size(self._window)
        self.aspect = fbw / fbh

        self.camera = LKGCamera(
            size=lkg_size,
            center=lkg_center,
            up=lkg_up,
            fov=fov,
            viewcone=lkg_viewcone,
            aspect=self.aspect,
            near=near,
            far=far
        )

        # These are used to allow for keyboard controls for the focus and offset (think depthiness from the RGBD tutorial)
        self.focus = 0.0
        self.offset = 1.0
        self.camera_distance = camera_distance

        self._input_adjustment_speed = 0.25

        # Bridge Setup
        self.bridge_ok = False
        final_width, final_height = width, height
        try:
            # this follows almost the same pattern as the `Displaying a Prerendered RGBD` tutorial.

            # Initialize the bridge class
            self.bridge = BridgeAPI()
            if not self.bridge.initialize("BridgePythonSample"):
                raise RuntimeError("Bridge initialize failed")

            # Bring up a window on the first looking glass display
            self.br_wnd = self.bridge.instance_window_gl(-1)

            # Throw an error if this fails
            if self.br_wnd == 0:
                raise RuntimeError("instance_window_gl returned 0")

            # If we do have a valid window we then need to retrieve the quilt settings for that 
            # looking glass display. You can use different settings but this is what we feel is 
            # a good mix of efficiency and visual quality
            asp, qw, qh, cols, rows = self.bridge.get_default_quilt_settings(self.br_wnd)
            self.br_aspect = float(asp)
            self.qw, self.qh, self.cols, self.rows = qw, qh, cols, rows

            # Here we do something a bit extra, we want to ensure that the 2D window is the same aspect ratio as the
            # looking glass display, so we will read the calibration for this particular display from bridge and 
            # use the display width and display height
            try:
                cal = self.bridge.get_calibration_gl(self.br_wnd)
                if cal and hasattr(cal, "screenW") and hasattr(cal, "screenH"):
                    dw, dh = cal.screenW, cal.screenH
                else:
                    dw, dh = (1920, 1080) if asp >= 1.0 else (1440, 2560)
            except Exception:
                dw, dh = (1920, 1080) if asp >= 1.0 else (1440, 2560)

            # then we will make sure the 2D window isn't too big
            final_width = max(320, min(800, dw // 3))
            final_height = max(240, min(800, dh // 3))

            glfw.set_window_size(self._window, final_width, final_height)

            # We resized the window and don't want it to be in a weird place on screen 
            # So this code below gets the main monitor information and will place the 
            # 2d window in the center of that monitor.
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            x = (mode.size.width - final_width) // 2
            y = (mode.size.height - final_height) // 2
            glfw.set_window_pos(self._window, x, y)

            # This function initializes the quilt framebuffer that we will be rendering to.
            self._init_quilt_buffers()
            print("Bridge ready!")
            print(f"Quilt settings: {qw}x{qh} ({cols}×{rows})", file=sys.stderr)
            print(f"Window size: {final_width}x{final_height}", file=sys.stderr)

            self.bridge_ok = True
        except Exception as e:
            print("Bridge disabled:", e, file=sys.stderr)
            self.br_aspect = self.aspect

        # Now that bridge has been initialized (or not) we make sure all the camera settings are correct
        # Just incase the window size or aspect changed.
        GL.glViewport(0, 0, final_width, final_height)
        self.aspect = final_width / final_height
        self.camera.aspect = self.aspect

        # This is how we store the meshes that we will be rendering to the quilt and 2d window.
        self._objects: List[Dict[str, Any]] = []

    # This is how objects are added to the renderer, each object is a mesh, a shader, and a model matrix.
    def add_object(self, mesh: "Mesh", shader: "Shader", model: Optional[np.ndarray] = None) -> int:
        # If we do not have a model matrix use a default matrix.
        if model is None:
            model = np.eye(4, dtype=np.float32)
        loc = GL.glGetUniformLocation(shader.id, "u_mvp")
        self._objects.append({"mesh": mesh, "shader": shader, "loc": loc, "model": model})
        return len(self._objects) - 1

    # Its INCREDIBLY useful to be able to change the offset and focus
    # used for rendering to the quilt, so we include a bit of extraneous code.
    # Technically you don't need it for this tutorial but its so important that
    # I didn't want to delete it.
    def _handle_keyboard_input(self, dt: float) -> None:
        fast = 10.0 if (
            glfw.get_key(self._window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(self._window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        ) else 1.0

        step = self._input_adjustment_speed * dt * fast

        # If the arrow keys are pressed change the focus or offset
        # Shift will increase the speed at which its changed.
        if glfw.get_key(self._window, glfw.KEY_UP) == glfw.PRESS:
            self.focus = min(10.0, self.focus + step)
            print(f"Focus: {self.focus:.2f}", file=sys.stderr)
        if glfw.get_key(self._window, glfw.KEY_DOWN) == glfw.PRESS:
            self.focus = max(-10.0, self.focus - step)
            print(f"Focus: {self.focus:.2f}", file=sys.stderr)
        if glfw.get_key(self._window, glfw.KEY_LEFT) == glfw.PRESS:
            self.offset = max(0.0, self.offset - step)
            print(f"Depthiness: {self.offset:.2f}", file=sys.stderr)
        if glfw.get_key(self._window, glfw.KEY_RIGHT) == glfw.PRESS:
            self.offset = min(5.0, self.offset + step)
            print(f"Depthiness: {self.offset:.2f}", file=sys.stderr)

        # If the esc key is pressed close the application
        if glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self._window, True)

    # To render to anything in opengl you need a framebuffer, this is a special texture that will be rendered to
    # a framebuffer is a bit more complex because you also need a depth buffer to ensure you draw the triangles
    # in the right order.
    # We want to pass the final rendered quilt as a texture to bridge so we will also need to create a texture reference.
    def _init_quilt_buffers(self) -> None:
        # This code below generates a texture using the quilt width and quilt height we got from bridge in the constructor.
        # This texture is only the color portion of the framebuffer we will be rendering to.
        self.quilt_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.quilt_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.qw, self.qh, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None
        )

        # This is the Render buffer it stores the data for the depth buffer that the framebuffer needs.
        self.depth_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth_rb)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, self.qw, self.qh)

        # Finally we create a framebuffer and tell it to use the quilt texture for the color, and the 
        # render buffer for the depth.
        self.quilt_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.quilt_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.quilt_tex, 0
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.depth_rb
        )

        # finally we make sure that the framebuffer is complete and working, if not throw an error.
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Quilt FBO incomplete")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # This function is used to draw all of the meshes to the current framebuffer with a particular camera matrix
    def _draw_objects(self, view: np.ndarray, proj: np.ndarray) -> None:
        for obj in self._objects:
            # this does the linear algebra needed to convert from the model space coordinates that the mesh
            # data is in into screen space coordinates. We will discuss this more in the LKGCamera section
            mvp = proj @ view @ obj["model"]
            # Activate the Shader for this particular mesh
            obj["shader"].use()
            # upload the camera matrix we calculated above into the GPU
            GL.glUniformMatrix4fv(obj["loc"], 1, GL.GL_TRUE, mvp)
            # Call the draw function on the mesh
            obj["mesh"].draw(GL.GL_TRIANGLES)

    # Ok, now that we have all the setup work done, we can get to the actual interesting part
    # This function is called once per frame and renders to both the 2d window and to the looking glass display
    def render_frame(self, dt: float = 0.016) -> None:
        # First we want to deal with any keyboard input
        self._handle_keyboard_input(dt)

        # Render to the 2D window
        self.render_2D()

        # If bridge is initialized and ready render to the looking glass display
        if self.bridge_ok:
            self.render_quilt()

        # Swap buffers takes the framebuffer which we renders to and actually displays it on the screen
        glfw.swap_buffers(self._window)

        # Poll window events such as keyboard and mouse input and any window buttons like the close or minimize buttons
        glfw.poll_events()

    # This is where we render to the 2d display.
    def render_2D(self):
        # We use the same camera to render the quilt and the 2d window just to ensure 
        # the output on the 2d window matches the looking glass fully. Here we get the camera
        # for the center view.
        center_view, center_proj = self.camera.compute_view_projection_matrices(
            0.5, False, self.offset, self.focus
        )

        # We need to ensure we are using the framebuffer for the 2d window NOT the quilt framebuffer
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # The viewport is the portion of the framebuffer that we will be rendering to
        # in this case we will be rendering to the full thing, which normally would be
        # the default, but because we change the viewport in the quilt rendering code
        # we need to reset it back to the full framebuffer here for the 2d window
        fbw, fbh = glfw.get_framebuffer_size(self._window)
        GL.glViewport(0, 0, fbw, fbh)
        
        # All that is left is to clear the framebuffer and draw all the objects!
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self._draw_objects(center_view, center_proj)

    # Ok we are finally here, rendering the quilt. 
    # This is very similar to the render_2D function above, but we are going to render once 
    # for each view in the quilt. 
    def render_quilt(self):
        # Here we want to bind the quilt framebuffer that we created
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.quilt_fbo)

        # Clear it
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Then we draw the quilt
        vw, vh = self.qw // self.cols, self.qh // self.rows
        total = self.cols * self.rows
        for y in range(self.rows):
            for x in range(self.cols):
                # This is the key piece that allows you to easily render the quilt
                # Essentially we are going to set the viewport to only render to the space 
                # where a single view exists on the quilt. 

                # We use the 
                GL.glViewport(x * vw, (self.rows - 1 - y) * vh, vw, vh)

                # We compute the view index based on the x and y coordinate 
                idx = y * self.cols + x

                # then we normalize the view index to get the direction for this view
                nrm = idx / (total - 1) if total > 1 else 0.5

                # finally we get the camera matrices for that particular view and draw
                v_mat, p_mat = self.camera.compute_view_projection_matrices(
                    nrm, True, self.offset, self.focus
                )
                self._draw_objects(v_mat, p_mat)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        # now that we have the quilt fully drawn we will render it to the looking glass using the bridge call.
        self.bridge.draw_interop_quilt_texture_gl(
            self.br_wnd,
            self.quilt_tex,
            PixelFormats.RGBA,
            self.qw,
            self.qh,
            self.cols,
            self.rows,
            self.br_aspect,
            1.0
        )

    # finally like the other classes we need to make sure that this class cleans up after itself.
    def close(self) -> None:
        if self.bridge_ok:
            try:
                GL.glDeleteFramebuffers(1, [self.quilt_fbo])
                GL.glDeleteTextures(1, [self.quilt_tex])
                GL.glDeleteRenderbuffers(1, [self.depth_rb])
            except Exception:
                pass
        if self._window:
            glfw.destroy_window(self._window)
            self._window = None
        glfw.terminate()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

class LKGCamera:
    def __init__(
        self,
        size: float = 3.0,
        center: Optional[np.ndarray] = None,
        up: Optional[np.ndarray] = None,
        fov: float = 14.0,
        viewcone: float = 40.0,
        aspect: float = 1.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> None:
        self.size = size
        self.center = center if center is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up = up if up is not None else np.array([0.0, -1.0, 0.0], dtype=np.float32)
        self.fov = fov
        self.viewcone = viewcone
        self.aspect = aspect
        self.near = near
        self.far = far

    # This is the function that we call in the renderer to get the camera matricies for any particular view
    # normalized view goes between 0 and 1 where 0 is the far left view and 1 is the far right view
    # invert flips the rendered image in the y axis
    # depthiness and focus work as you would expect the same as the RGBD example.
    def compute_view_projection_matrices(
        self,
        normalized_view: float,
        invert: bool,
        depthiness: float,
        focus: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # The size sets the size of the volume that will be rendered, 
        # we then divide this by the tangent of the fov to get
        # the camera_distance. This is the distance that the camera 
        # should away from the subject to create an appropriately sized volume.
        # In our camera rail analogy this is the distance between the lens of 
        # the camera and the subject
        camera_distance = self.size / math.tan(math.radians(self.fov))

        # camera_offset here is the maximum travel distance, we need to travel the correct distance
        # to correspond to the viewcone of the looking glass display.
        # In our camera rail analogy this is the distance that the camera travels left to right
        camera_offset = camera_distance * math.tan(math.radians(self.viewcone))

        # For the actual camera matrix computation we need to normalize the view to be between -0.5 and 0.5
        # and multiply it by the camera_offset. 
        view_from_center = normalized_view - 0.5
        offset = -(view_from_center) * depthiness * camera_offset

        # This computes the view matrix for this particular offset and camera_distance
        # we will go into detail in the function itself
        view = self._view_matrix(offset, camera_distance, invert)

        # This is just a standard projection matrix, but we need to create it here
        # because we will be tweaking it slightly
        proj = self._projection_matrix()

        # If we are inverted the focus needs to be flipped
        if invert:
            focus = -focus

        # Remember in our camera rail analogy how the real world capture creates an almost
        # accurate volumetric render, this shift to the camera frustrum basically corrects
        # for the fact that we are moving the camera position but we want the actual camera
        # plane to remain in the same place.
        # We also need to do some weirdness because the np arrays we are using for the camera
        # matricies are in the wrong order for opengl.
        # Write to row-major slot (0,2) so that, after the transpose in glUniformMatrix4fv,
        # the value lands in column-major M31.
        frustum_shift = view_from_center * focus
        proj[0, 2] += (offset * 2.0 / (self.size * self.aspect)) + frustum_shift

        # Thats it, the hardest math-y-est part is done!
        return view, proj

    # This creates a camera based on the offset from the center, and the camera distance we calculated.
    def _view_matrix(self, offset: float, camera_distance: float, invert: bool) -> np.ndarray:
        # this is all normal camera math stuff
        f = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        u_base = self.up.copy()
        if invert:
            u_base[1] = -u_base[1]
        u = u_base / np.linalg.norm(u_base)
        s = np.cross(f, u)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)

        # this creates a camera that is camera_distance away from the subject
        # and shifted left / right based on the input offset value
        m = np.eye(4, dtype=np.float32)
        m[0, 0], m[1, 0], m[2, 0] = s
        m[0, 1], m[1, 1], m[2, 1] = u
        m[0, 2], m[1, 2], m[2, 2] = -f
        m[0, 3] = offset
        m[2, 3] = -camera_distance

        if not invert:
            m = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32) @ m
        return m

    # This creates a standard projection matrix based on the fov and aspect
    def _projection_matrix(self) -> np.ndarray:
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        n, fp = self.near, self.far
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / self.aspect
        m[1, 1] = f
        m[2, 2] = (fp + n) / (n - fp)
        m[2, 3] = (2 * fp * n) / (n - fp)
        m[3, 2] = -1.0
        return m

def main() -> None:
    # Create a renderer, a mesh, and a shader
    # We have to create the renderer first because it sets up opengl
    renderer = Render()
    mesh   = Mesh()
    shader = Shader()

    # Add the mesh and shader to the renderer as an object
    handle = renderer.add_object(mesh, shader)

    # variables to keep track of the cube rotation
    rx = ry = rz = 0.0
    auto_rx, auto_ry, auto_rz = 0.07, 0.1, 0.09

    last_time = time.time()

    # while the 2d window is open
    while not glfw.window_should_close(renderer._window):
        # update the rotation of the mesh to make the example more interesting
        now = time.time()
        dt = now - last_time
        last_time = now

        rx += auto_rx * dt
        ry += auto_ry * dt
        rz += auto_rz * dt

        # update the model matrix based on the rotations
        renderer._objects[handle]["model"] = Mesh.euler_xyz(rx, ry, rz)
        
        # render a frame!
        renderer.render_frame(dt)

# this is some python boiler plate to ensure we only run the 
# main function when this file is executed instead of any time this file is referenced
if __name__ == "__main__":
    main()