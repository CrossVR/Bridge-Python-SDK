#!/usr/bin/env python3
# Mesh.py – robust, driver-safe upload
import numpy as np
import ctypes
from OpenGL.GL import *

class Mesh:
    """
    Encapsulates a VAO/VBO mesh.  Provide interleaved vertex data and attribute
    layout, then call draw() to render.
    """
    def __init__(self, vertices: np.ndarray, attribs: list[tuple]):
        """
        vertices: flat numpy array of vertex data (dtype=np.float32).
        attribs:  list of attribute specs, each a tuple
                  (location, size, type, normalized, stride, offset)
        """
        # ---- prepare data --------------------------------------------------
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self._vertices_ctypes = (ctypes.c_float * vertices.size).from_buffer(vertices)
        self.vertex_count = vertices.nbytes // attribs[0][4]  # bytes / stride

        # ---- VAO -----------------------------------------------------------
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # ---- VBO -----------------------------------------------------------
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(
            GL_ARRAY_BUFFER,
            vertices.nbytes,
            self._vertices_ctypes,           # typed buffer stays alive via self
            GL_STATIC_DRAW
        )

        # ---- attribute layout ---------------------------------------------
        for loc, size, typ, norm, stride, offset in attribs:
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(
                loc, size, typ, norm, stride, ctypes.c_void_p(offset)
            )

        # ---- tidy up -------------------------------------------------------
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def draw(self, mode: int = GL_TRIANGLES) -> None:
        """Bind the VAO and issue a draw call (default: triangles)."""
        glBindVertexArray(self.vao)
        glDrawArrays(mode, 0, self.vertex_count)
        glBindVertexArray(0)

    def __del__(self):
        """Delete GL objects when the instance is garbage-collected."""
        try:
            glDeleteBuffers(1, [self.vbo])
            glDeleteVertexArrays(1, [self.vao])
        except Exception:
            pass
