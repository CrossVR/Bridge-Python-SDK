#!/usr/bin/env python3
# Window.py
import glfw
from OpenGL.GL import *

class Window:
    def __init__(self, width: int, height: int, title: str,
                 gl_major: int = 4, gl_minor: int = 6, core_profile: bool = True):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, gl_major)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, gl_minor)
        if core_profile:
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        self._width = width
        self._height = height
        self._window = glfw.create_window(width, height, title, None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self._window)

    def should_close(self) -> bool:
        return glfw.window_should_close(self._window)

    def swap_buffers(self) -> None:
        glfw.swap_buffers(self._window)

    def poll_events(self) -> None:
        glfw.poll_events()

    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0) -> None:
        glClearColor(r, g, b, a)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def set_resize_callback(self, callback) -> None:
        def _cb(window, w, h):
            self._width = w
            self._height = h
            callback(window, w, h)
        glfw.set_framebuffer_size_callback(self._window, _cb)

    def get_size(self) -> tuple[int, int]:
        return self._width, self._height

    def framebuffer_size(self) -> tuple[int, int]:
        return glfw.get_framebuffer_size(self._window)

    def terminate(self) -> None:
        glfw.destroy_window(self._window)
        glfw.terminate()
