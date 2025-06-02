#!/usr/bin/env python3
# Shader.py

import os
import sys
import math
import argparse
import numpy as np
from PIL import Image
import glfw
from OpenGL.GL import *
import ctypes

class Shader:
    def __init__(self, vertex_src: str, fragment_src: str):
        self.id = glCreateProgram()
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

    def _compile(self, src: str, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, src)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            log = glGetShaderInfoLog(shader).decode()
            glDeleteShader(shader)
            raise RuntimeError(f"Shader compile failed:\n{log}")
        return shader

    def use(self):
        glUseProgram(self.id)

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
