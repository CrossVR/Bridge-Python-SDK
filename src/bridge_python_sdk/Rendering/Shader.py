#!/usr/bin/env python3
# Shader.py

import numpy as np
from OpenGL.GL import *


class Shader:
    def __init__(self, vertex_src: str, fragment_src: str):
        self.id = glCreateProgram()
        self._numbered_srcs = {}
        vert = self._compile(vertex_src, GL_VERTEX_SHADER)
        frag = self._compile(fragment_src, GL_FRAGMENT_SHADER)
        glAttachShader(self.id, vert)
        glAttachShader(self.id, frag)
        glLinkProgram(self.id)
        if not glGetProgramiv(self.id, GL_LINK_STATUS):
            log = glGetProgramInfoLog(self.id).decode()
            glDeleteProgram(self.id)
            glDeleteShader(vert)
            glDeleteShader(frag)
            raise RuntimeError(f"Program link failed:\n{log}")
        glDeleteShader(vert)
        glDeleteShader(frag)

    # --------------------------------------------------------------------- internal
    def _compile(self, src: str, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, src)
        numbered_src = '\n'.join(f"{i + 1:4d}: {line}"
                                 for i, line in enumerate(src.splitlines()))
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            log = glGetShaderInfoLog(shader).decode()
            glDeleteShader(shader)
            type_name = {GL_VERTEX_SHADER: "vertex",
                         GL_FRAGMENT_SHADER: "fragment"}.get(shader_type,
                                                             str(shader_type))
            raise RuntimeError(f"{type_name.capitalize()} shader compile failed:\n"
                               f"{log}\nSource with line numbers:\n{numbered_src}")
        self._numbered_srcs[shader_type] = numbered_src
        return shader

    # --------------------------------------------------------------------- program use
    def use(self):
        glUseProgram(self.id)

    # --------------------------------------------------------------------- simple setters
    def set_int(self, name: str, value: int):
        loc = glGetUniformLocation(self.id, name)
        glUniform1i(loc, value)

    def set_float(self, name: str, value: float):
        loc = glGetUniformLocation(self.id, name)
        glUniform1f(loc, value)

    def set_vec2(self, name: str, x: float, y: float):
        loc = glGetUniformLocation(self.id, name)
        glUniform2f(loc, x, y)

    def set_vec3(self, name: str, x: float, y: float, z: float):
        loc = glGetUniformLocation(self.id, name)
        glUniform3f(loc, x, y, z)

    def set_vec4(self, name: str, x: float, y: float, z: float, w: float):
        loc = glGetUniformLocation(self.id, name)
        glUniform4f(loc, x, y, z, w)

    def set_uniform(self, name: str, *value):
        """
        Generic scalar/vector uniform setter—unchanged from original.
        Accepts either separate components or a single iterable.
        """
        loc = glGetUniformLocation(self.id, name)
        if loc == -1:
            print("Vertex shader source:\n", self._numbered_srcs.get(GL_VERTEX_SHADER, "<none>"))
            print("Fragment shader source:\n", self._numbered_srcs.get(GL_FRAGMENT_SHADER, "<none>"))
            raise ValueError(f"Uniform '{name}' not found in program")

        if len(value) == 1 and isinstance(value[0], (tuple, list, np.ndarray)):
            value = tuple(value[0])

        try:
            length = len(value)
            if length == 1 and isinstance(value[0], (int, np.integer)):
                glUniform1i(loc, int(value[0]))
            elif length == 1:
                glUniform1f(loc, float(value[0]))
            elif length == 2:
                glUniform2f(loc, float(value[0]), float(value[1]))
            elif length == 3:
                glUniform3f(loc, float(value[0]), float(value[1]), float(value[2]))
            elif length == 4:
                glUniform4f(loc, float(value[0]), float(value[1]),
                            float(value[2]), float(value[3]))
            else:
                raise ValueError(f"Unsupported uniform vector length {length}")
        except Exception as e:
            print("Vertex shader source:\n", self._numbered_srcs.get(GL_VERTEX_SHADER, "<none>"))
            print("Fragment shader source:\n", self._numbered_srcs.get(GL_FRAGMENT_SHADER, "<none>"))
            raise

    # --------------------------------------------------------------------- matrix setters (NEW)
    def set_uniform_matrix(self, name: str, matrix, transpose: bool = True):
        """
        Upload a 4×4 float matrix uniform.

        Parameters
        ----------
        name : str
            Uniform variable name.
        matrix : array-like, shape (4, 4)
            Matrix data (row-major from NumPy).  OpenGL expects column-major,
            so transpose=True by default.
        transpose : bool
            Whether to transpose the matrix when sending to GL.
        """
        loc = glGetUniformLocation(self.id, name)
        if loc == -1:
            raise ValueError(f"Uniform '{name}' not found in program")
        mat = np.asarray(matrix, dtype=np.float32)
        if mat.shape != (4, 4):
            raise ValueError(f"Matrix uniform '{name}' must be 4×4")
        glUniformMatrix4fv(loc, 1,
                           GL_TRUE if transpose else GL_FALSE,
                           mat)

    def set_uniform_matrix3(self, name: str, matrix3, transpose: bool = True):
        """
        Upload a 3×3 float matrix uniform (e.g., normal matrix).
        """
        loc = glGetUniformLocation(self.id, name)
        if loc == -1:
            raise ValueError(f"Uniform '{name}' not found in program")
        mat = np.asarray(matrix3, dtype=np.float32)
        if mat.shape != (3, 3):
            raise ValueError(f"Matrix uniform '{name}' must be 3×3")
        glUniformMatrix3fv(loc, 1,
                           GL_TRUE if transpose else GL_FALSE,
                           mat)
