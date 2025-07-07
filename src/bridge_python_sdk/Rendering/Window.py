#!/usr/bin/env python3
# Window.py
import sys
import glfw
from OpenGL.GL import *
from typing import Callable, Optional, Tuple

class Window:
    def __init__(self, width: int, height: int, title: str,
                 gl_major: int = 4, gl_minor: int = 6, core_profile: bool = True):
        if sys.platform == "darwin":
            if (gl_major, gl_minor) > (4, 1):
                gl_major, gl_minor = 4, 1

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, gl_major)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, gl_minor)
        if core_profile:
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if sys.platform == "darwin":
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

        self._width = width
        self._height = height
        self._window = glfw.create_window(width, height, title, None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)  # Enable v-sync by default

        # Print context info for verification
        print("OpenGL :", glGetString(GL_VERSION).decode())
        print("GLSL   :", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())

        # Track framebuffer resize
        def _resize_callback(window, w, h):
            self._width, self._height = glfw.get_window_size(self._window)
        glfw.set_framebuffer_size_callback(self._window, _resize_callback)

    @property
    def handle(self) -> int:
        return self._window

    @property
    def window(self) -> int:
        return self._window

    def should_close(self) -> bool:
        return glfw.window_should_close(self._window)

    def swap_buffers(self) -> None:
        glfw.swap_buffers(self._window)

    def poll_events(self) -> None:
        glfw.poll_events()

    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 1.0) -> None:
        glClearColor(r, g, b, a)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def set_resize_callback(self, callback: Callable) -> None:
        def _cb(window, w, h):
            self._width, self._height = glfw.get_window_size(self._window)
            callback(window, w, h)
        glfw.set_framebuffer_size_callback(self._window, _cb)

    def get_size(self) -> Tuple[int, int]:
        return glfw.get_window_size(self._window)

    def framebuffer_size(self) -> Tuple[int, int]:
        return glfw.get_framebuffer_size(self._window)

    def set_size(self, width: int, height: int) -> None:
        glfw.set_window_size(self._window, width, height)
        self._width = width
        self._height = height

    def set_position(self, x: int, y: int) -> None:
        glfw.set_window_pos(self._window, x, y)

    def get_position(self) -> Tuple[int, int]:
        return glfw.get_window_pos(self._window)

    def center_on_screen(self) -> None:
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        width, height = self.get_size()
        x = (mode.size.width - width) // 2
        y = (mode.size.height - height) // 2
        self.set_position(x, y)

    def set_title(self, title: str) -> None:
        glfw.set_window_title(self._window, title)

    def set_vsync(self, enabled: bool) -> None:
        glfw.swap_interval(1 if enabled else 0)

    # ===== Keyboard Input =====
    def is_key_pressed(self, key: int) -> bool:
        return glfw.get_key(self._window, key) == glfw.PRESS

    def is_key_released(self, key: int) -> bool:
        return glfw.get_key(self._window, key) == glfw.RELEASE

    def get_key(self, key: int) -> int:
        return glfw.get_key(self._window, key)

    def set_key_callback(self, callback: Callable) -> None:
        glfw.set_key_callback(self._window, callback)

    # ===== Mouse Input =====
    def get_mouse_pos(self) -> Tuple[float, float]:
        return glfw.get_cursor_pos(self._window)

    def set_mouse_pos(self, x: float, y: float) -> None:
        glfw.set_cursor_pos(self._window, x, y)

    def is_mouse_button_pressed(self, button: int) -> bool:
        return glfw.get_mouse_button(self._window, button) == glfw.PRESS

    def get_mouse_button(self, button: int) -> int:
        return glfw.get_mouse_button(self._window, button)

    def set_mouse_button_callback(self, callback: Callable) -> None:
        glfw.set_mouse_button_callback(self._window, callback)

    def set_cursor_pos_callback(self, callback: Callable) -> None:
        glfw.set_cursor_pos_callback(self._window, callback)

    def set_scroll_callback(self, callback: Callable) -> None:
        glfw.set_scroll_callback(self._window, callback)

    def hide_cursor(self) -> None:
        glfw.set_input_mode(self._window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

    def show_cursor(self) -> None:
        glfw.set_input_mode(self._window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def disable_cursor(self) -> None:
        glfw.set_input_mode(self._window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    # ===== Window State =====
    def maximize(self) -> None:
        glfw.maximize_window(self._window)

    def minimize(self) -> None:
        glfw.iconify_window(self._window)

    def restore(self) -> None:
        glfw.restore_window(self._window)

    def is_maximized(self) -> bool:
        return glfw.get_window_attrib(self._window, glfw.MAXIMIZED) == glfw.TRUE

    def is_minimized(self) -> bool:
        return glfw.get_window_attrib(self._window, glfw.ICONIFIED) == glfw.TRUE

    def is_focused(self) -> bool:
        return glfw.get_window_attrib(self._window, glfw.FOCUSED) == glfw.TRUE

    def focus(self) -> None:
        glfw.focus_window(self._window)

    def set_resizable(self, resizable: bool) -> None:
        glfw.set_window_attrib(self._window, glfw.RESIZABLE, glfw.TRUE if resizable else glfw.FALSE)

    def set_always_on_top(self, on_top: bool) -> None:
        glfw.set_window_attrib(self._window, glfw.FLOATING, glfw.TRUE if on_top else glfw.FALSE)

    # ===== Cleanup =====
    def terminate(self) -> None:
        if self._window:
            glfw.destroy_window(self._window)
            self._window = None
        glfw.terminate()

    def __del__(self):
        if hasattr(self, '_window') and self._window:
            glfw.destroy_window(self._window)

    # ===== Context Management =====
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()
