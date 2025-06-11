#!/usr/bin/env python3
# Mesh.py
import numpy as np
from OpenGL.GL import *
import ctypes

class Mesh:
    """
    Encapsulates a VAO/VBO mesh. Provide interleaved vertex data and attribute layout,
    and then call draw() to render.
    """
    def __init__(self, vertices: np.ndarray, attribs: list[tuple]):
        """
        vertices: flat numpy array of vertex data (dtype=np.float32).
        attribs: list of attribute specs, each a tuple
            (location, size, type, normalized, stride, offset)
            where:
              - location: the shader layout location
              - size: number of components (e.g. 2 for vec2)
              - type: GL_FLOAT, GL_INT, etc.
              - normalized: True/False
              - stride: total byte size of one vertex
              - offset: byte offset of this attribute within the vertex
        """
        self.vertex_count = len(vertices) * vertices.itemsize // attribs[0][4]
        # generate and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        # generate VBO and upload data
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        # set up each attribute
        for loc, size, typ, norm, stride, offset in attribs:
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(
                loc, size, typ, norm,
                stride,
                ctypes.c_void_p(offset)
            )
        # unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def draw(self, mode=GL_TRIANGLE_STRIP):
        """
        Bind VAO and issue draw call.
        mode: GL_TRIANGLES, GL_TRIANGLE_STRIP, etc.
        """
        glBindVertexArray(self.vao)
        glDrawArrays(mode, 0, self.vertex_count)
        glBindVertexArray(0)

    def __del__(self):
        """
        Cleanup GL objects when garbage-collected.
        """
        try:
            glDeleteBuffers(1, [self.vbo])
            glDeleteVertexArrays(1, [self.vao])
        except Exception:
            pass
