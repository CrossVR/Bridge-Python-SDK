#!/usr/bin/env python3
# RotatingCube.py  –  complete file, no wildcard imports
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
    vUV = aPos.xz * 0.5 + 0.5;   // simple planar UV from XZ
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

def rotation_y(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[ c,0, s,0],
                     [ 0,1, 0,0],
                     [-s,0, c,0],
                     [ 0,0, 0,1]], dtype=np.float32)

def main() -> None:
    renderer = Render()
    mesh = Mesh(cube_vertices, attribs)
    shader = Shader(vertex_shader_src, fragment_shader_src)
    handle = renderer.add_object(mesh, shader)

    start = time.time()
    last_time = start
    while not renderer.should_close():
        now = time.time()
        delta = now - last_time
        last_time = now
        renderer.update_model(handle, rotation_y(now - start))
        renderer.render_frame(delta)

if __name__ == "__main__":
    main()
