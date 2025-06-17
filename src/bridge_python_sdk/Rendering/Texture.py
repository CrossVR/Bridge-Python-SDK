#!/usr/bin/env python3
# Texture.py
import os
import sys
import math
import argparse
import numpy as np
from PIL import Image
import glfw
from OpenGL.GL import *
import ctypes

class Texture2D:
    def __init__(self, path: str):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        data = np.frombuffer(img.tobytes(), dtype=np.uint8)
        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def bind(self, unit: int):
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.id)

class Texture3D:
    def __init__(self, paths: list):
        imgs = [Image.open(p).convert("RGB") for p in paths]
        self.count = len(imgs)
        w, h = imgs[0].size
        array = np.stack([np.frombuffer(img.tobytes(), dtype=np.uint8).reshape((h, w, 3)) for img in imgs], axis=0)
        data = array.tobytes()
        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB, w, h, self.count, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)

    def bind(self, unit: int):
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_3D, self.id)

class TextureCube:
    def __init__(self, paths: list):
        if len(paths) != 6:
            raise ValueError("TextureCube requires exactly six image paths")
        imgs = [Image.open(p).convert("RGB") for p in paths]
        w, h = imgs[0].size
        for img in imgs:
            if img.size != (w, h):
                raise ValueError("All cube map faces must have the same dimensions")
        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        targets = [
            GL_TEXTURE_CUBE_MAP_POSITIVE_X,
            GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
            GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
            GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
            GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
            GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
        ]
        for img, target in zip(imgs, targets):
            data = np.frombuffer(img.tobytes(), dtype=np.uint8)
            glTexImage2D(target, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    def bind(self, unit: int):
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.id)
