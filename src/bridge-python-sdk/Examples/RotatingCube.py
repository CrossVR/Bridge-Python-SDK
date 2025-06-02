#!/usr/bin/env python3
# Render.py  –  complete file, no wildcard imports
import math
import time
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from OpenGL import GL

from Rendering.Shader import Shader
from Rendering.Mesh import Mesh
from Rendering.Render import Render

# ------------------------------------------------------------ cube geometry

# Fixed cube vertices - only position data (3 floats per vertex)
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

# Attribute layout: only position (3 floats)
stride = 3 * 4  # 3 floats * 4 bytes per float
attribs = [(0, 3, GL.GL_FLOAT, False, stride, 0)]

vertex_shader_src = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 u_mvp;
void main(){ gl_Position = u_mvp * vec4(aPos,1.0); }"""

fragment_shader_src = """
#version 330 core
out vec4 FragColor;
void main(){ FragColor = vec4(0.3,0.6,1.0,1.0); }"""


def rotation_y(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[ c,0, s,0],
                     [ 0,1, 0,0],
                     [-s,0, c,0],
                     [ 0,0, 0,1]], dtype=np.float32)


def main() -> None:
    renderer = Render(debug=True)
    mesh = Mesh(cube_vertices, attribs)
    shader = Shader(vertex_shader_src, fragment_shader_src)
    handle = renderer.add_object(mesh, shader)

    start = time.time()
    while not renderer.should_close():
        renderer.update_model(handle, rotation_y(time.time() - start))
        renderer.render_frame()


if __name__ == "__main__":
    main()
