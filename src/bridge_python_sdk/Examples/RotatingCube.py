#!/usr/bin/env python3
# RotatingCube.py  –  WASD + QE + mouse drag, auto-spin pauses after user input
import math
import time
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from OpenGL import GL
import glfw

from Rendering.Shader import Shader
from Rendering.Mesh import Mesh
from Rendering.Render import Render

# ------------------------------------------------------------ cube geometry
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

stride = 3 * 4  # position only
attribs = [(0, 3, GL.GL_FLOAT, False, stride, 0)]

vertex_shader_src = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 u_mvp;
out vec2 vUV;
void main(){
    gl_Position = u_mvp * vec4(aPos,1.0);
    vUV = aPos.xz * 0.5 + 0.5;
}
"""

fragment_shader_src = """
#version 330 core
in vec2 vUV;
out vec4 FragColor;
void main(){
    vec3 col = vec3(vUV, 1.0 - vUV.x);
    FragColor = vec4(col,1.0);
}
"""

# ------------------------------------------------------------ math helpers
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

# ------------------------------------------------------------ application
def main() -> None:
    renderer = Render(debug=True)
    mesh = Mesh(cube_vertices, attribs)
    shader = Shader(vertex_shader_src, fragment_shader_src)
    handle = renderer.add_object(mesh, shader)

    rx = ry = rz = 0.0
    auto_rx, auto_ry, auto_rz = 0.7, 1.1, 0.9   # rad / sec
    key_speed = 2.5                              # rad / sec
    mouse_sens = 0.01                            # rad / pixel
    auto_pause = 3.0                             # seconds

    last_mouse = None
    start_time = time.time()
    last_time = start_time
    last_input_time = start_time

    while not renderer.should_close():
        now = time.time()
        dt = now - last_time
        last_time = now
        user_input = False

        # --- keyboard input (WASD + QE)
        if renderer.window.is_key_pressed(glfw.KEY_W):
            rx += key_speed * dt
            user_input = True
        if renderer.window.is_key_pressed(glfw.KEY_S):
            rx -= key_speed * dt
            user_input = True
        if renderer.window.is_key_pressed(glfw.KEY_A):
            ry += key_speed * dt
            user_input = True
        if renderer.window.is_key_pressed(glfw.KEY_D):
            ry -= key_speed * dt
            user_input = True
        if renderer.window.is_key_pressed(glfw.KEY_Q):
            rz += key_speed * dt
            user_input = True
        if renderer.window.is_key_pressed(glfw.KEY_E):
            rz -= key_speed * dt
            user_input = True

        # --- mouse drag input (left button)
        mb_pressed = renderer.window.is_mouse_button_pressed(glfw.MOUSE_BUTTON_LEFT)
        mx, my = renderer.window.get_mouse_pos()
        if mb_pressed:
            if last_mouse is not None:
                dx = mx - last_mouse[0]
                dy = my - last_mouse[1]
                ry += dx * mouse_sens
                rx += dy * mouse_sens
                user_input = True
            last_mouse = (mx, my)
        else:
            last_mouse = None

        if user_input:
            last_input_time = now

        # --- automatic spin (only if no input for auto_pause seconds)
        if now - last_input_time > auto_pause:
            rx += auto_rx * dt
            ry += auto_ry * dt
            rz += auto_rz * dt

        renderer.update_model(handle, euler_xyz(rx, ry, rz))
        renderer.render_frame(dt)

if __name__ == "__main__":
    main()
