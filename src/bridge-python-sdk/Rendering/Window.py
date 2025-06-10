#!/usr/bin/env python3
# Window.py
import glfw
from OpenGL.GL import *
from typing import Callable, Optional, Tuple

class Window:
    def __init__(self, width: int, height: int, title: str,
                 gl_major: int = 4, gl_minor: int = 6, core_profile: bool = True):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Set OpenGL context hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, gl_major)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, gl_minor)
        if core_profile:
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        
        # Additional useful hints
        glfw.window_hint(glfw.SAMPLES, 4)  # 4x MSAA
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        
        self._width = width
        self._height = height
        self._window = glfw.create_window(width, height, title, None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self._window)
        
        # Enable vsync by default
        glfw.swap_interval(1)
        
        # Set up resize callback to track actual size
        def _resize_callback(window, w, h):
            self._width, self._height = glfw.get_window_size(self._window)
        glfw.set_framebuffer_size_callback(self._window, _resize_callback)

    @property
    def handle(self) -> int:
        """Get the GLFW window handle for direct access"""
        return self._window

    @property
    def window(self) -> int:
        """Alias for handle - for compatibility"""
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
        """Get window size in screen coordinates"""
        return glfw.get_window_size(self._window)

    def framebuffer_size(self) -> Tuple[int, int]:
        """Get framebuffer size in pixels"""
        return glfw.get_framebuffer_size(self._window)

    def set_size(self, width: int, height: int) -> None:
        """Resize the window"""
        glfw.set_window_size(self._window, width, height)
        self._width = width
        self._height = height

    def set_position(self, x: int, y: int) -> None:
        """Set window position"""
        glfw.set_window_pos(self._window, x, y)

    def get_position(self) -> Tuple[int, int]:
        """Get window position"""
        return glfw.get_window_pos(self._window)

    def center_on_screen(self) -> None:
        """Center the window on the primary monitor"""
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        width, height = self.get_size()
        x = (mode.size.width - width) // 2
        y = (mode.size.height - height) // 2
        self.set_position(x, y)

    def set_title(self, title: str) -> None:
        """Set window title"""
        glfw.set_window_title(self._window, title)

    def set_vsync(self, enabled: bool) -> None:
        """Enable or disable vertical sync"""
        glfw.swap_interval(1 if enabled else 0)

    # ===== Keyboard Input =====
    def is_key_pressed(self, key: int) -> bool:
        """Check if a key is currently pressed"""
        return glfw.get_key(self._window, key) == glfw.PRESS

    def is_key_released(self, key: int) -> bool:
        """Check if a key is currently released"""
        return glfw.get_key(self._window, key) == glfw.RELEASE

    def get_key(self, key: int) -> int:
        """Get key state (PRESS, RELEASE, or REPEAT)"""
        return glfw.get_key(self._window, key)

    def set_key_callback(self, callback: Callable) -> None:
        """Set keyboard callback function"""
        glfw.set_key_callback(self._window, callback)

    # ===== Mouse Input =====
    def get_mouse_pos(self) -> Tuple[float, float]:
        """Get mouse cursor position"""
        return glfw.get_cursor_pos(self._window)

    def set_mouse_pos(self, x: float, y: float) -> None:
        """Set mouse cursor position"""
        glfw.set_cursor_pos(self._window, x, y)

    def is_mouse_button_pressed(self, button: int) -> bool:
        """Check if a mouse button is currently pressed"""
        return glfw.get_mouse_button(self._window, button) == glfw.PRESS

    def get_mouse_button(self, button: int) -> int:
        """Get mouse button state"""
        return glfw.get_mouse_button(self._window, button)

    def set_mouse_button_callback(self, callback: Callable) -> None:
        """Set mouse button callback function"""
        glfw.set_mouse_button_callback(self._window, callback)

    def set_cursor_pos_callback(self, callback: Callable) -> None:
        """Set mouse movement callback function"""
        glfw.set_cursor_pos_callback(self._window, callback)

    def set_scroll_callback(self, callback: Callable) -> None:
        """Set mouse scroll callback function"""
        glfw.set_scroll_callback(self._window, callback)

    def hide_cursor(self) -> None:
        """Hide the mouse cursor"""
        glfw.set_input_mode(self._window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

    def show_cursor(self) -> None:
        """Show the mouse cursor"""
        glfw.set_input_mode(self._window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def disable_cursor(self) -> None:
        """Disable cursor (for FPS-style camera control)"""
        glfw.set_input_mode(self._window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    # ===== Window State =====
    def maximize(self) -> None:
        """Maximize the window"""
        glfw.maximize_window(self._window)

    def minimize(self) -> None:
        """Minimize the window"""
        glfw.iconify_window(self._window)

    def restore(self) -> None:
        """Restore the window from minimized/maximized state"""
        glfw.restore_window(self._window)

    def is_maximized(self) -> bool:
        """Check if window is maximized"""
        return glfw.get_window_attrib(self._window, glfw.MAXIMIZED) == glfw.TRUE

    def is_minimized(self) -> bool:
        """Check if window is minimized"""
        return glfw.get_window_attrib(self._window, glfw.ICONIFIED) == glfw.TRUE

    def is_focused(self) -> bool:
        """Check if window has focus"""
        return glfw.get_window_attrib(self._window, glfw.FOCUSED) == glfw.TRUE

    def focus(self) -> None:
        """Give focus to the window"""
        glfw.focus_window(self._window)

    def set_resizable(self, resizable: bool) -> None:
        """Set whether the window can be resized"""
        glfw.set_window_attrib(self._window, glfw.RESIZABLE, glfw.TRUE if resizable else glfw.FALSE)

    def set_always_on_top(self, on_top: bool) -> None:
        """Set whether the window should stay on top"""
        glfw.set_window_attrib(self._window, glfw.FLOATING, glfw.TRUE if on_top else glfw.FALSE)

    # ===== Cleanup =====
    def terminate(self) -> None:
        """Clean up and terminate"""
        if self._window:
            glfw.destroy_window(self._window)
            self._window = None
        glfw.terminate()

    def __del__(self):
        """Destructor to ensure cleanup"""
        if hasattr(self, '_window') and self._window:
            glfw.destroy_window(self._window)

    # ===== Context Management =====
    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.terminate()
