# Render.py  – complete, ready to drop in
import math
import sys
import ctypes
from typing import Optional, Dict, Any, List

import numpy as np
from OpenGL import GL
from OpenGL.raw.GL.VERSION.GL_4_3 import glDebugMessageCallback as glDbgCB_Core
from OpenGL.raw.GL.VERSION.GL_4_3 import GLDEBUGPROC as GLDEBUGPROC_Core
from OpenGL.raw.GL.ARB.debug_output import glDebugMessageCallbackARB as glDbgCB_ARB
from OpenGL.raw.GL.ARB.debug_output import GLDEBUGPROCARB as GLDEBUGPROC_ARB

from BridgeApi import BridgeAPI, PixelFormats
from .Window import Window
from .Shader import Shader
from .Mesh import Mesh
from .LKGCamera import LKGCamera


class Render:
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        title: str = "",
        fov: float = 14.0,
        near: float = 0.5,
        far: float = 1000.0,
        debug: bool = False,
        camera_distance: float = 10.0,
        lkg_size: float = 3.0,
        lkg_center: Optional[np.ndarray] = None,
        lkg_up: Optional[np.ndarray] = None,
        lkg_viewcone: float = 40.0,
        lkg_invert: bool = True,
    ) -> None:
        self.debug = debug
        self.window = Window(width, height, title)
        if self.debug:
            self._enable_khr_debug()
            print("GL_VERSION:", GL.glGetString(GL.GL_VERSION).decode(), file=sys.stderr)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glEnable(GL.GL_DEPTH_TEST)

        # Interactive parameters
        self.focus = 0.0
        self.offset = 1.0
        self._input_adjustment_speed = 0.25

        # Aspect placeholder
        self.aspect = width / height

        # LKG camera
        self.camera = LKGCamera(
            size=lkg_size,
            center=lkg_center,
            up=lkg_up,
            fov=fov,
            viewcone=lkg_viewcone,
            aspect=self.aspect,
            near=near,
            far=far,
        )
        self.lkg_invert = lkg_invert
        self.camera_distance = camera_distance

        # ---------------- Bridge setup ----------------
        self.bridge_ok = False
        final_width, final_height = width, height
        try:
            self.bridge = BridgeAPI()
            if not self.bridge.initialize("BridgePythonSample"):
                raise RuntimeError("Bridge initialize failed")
            self.br_wnd = self.bridge.instance_window_gl(-1)
            if self.br_wnd == 0:
                raise RuntimeError("instance_window_gl returned 0")

            asp, qw, qh, cols, rows = self.bridge.get_default_quilt_settings(self.br_wnd)
            self.br_aspect = float(asp)
            self.qw, self.qh, self.cols, self.rows = qw, qh, cols, rows

            try:
                cal = self.bridge.get_calibration_gl(self.br_wnd)
                if cal and hasattr(cal, "screenW") and hasattr(cal, "screenH"):
                    dw, dh = cal.screenW, cal.screenH
                else:
                    dw, dh = (1920, 1080) if asp >= 1.0 else (1440, 2560)
            except Exception:
                dw, dh = (1920, 1080) if asp >= 1.0 else (1440, 2560)

            final_width = max(320, min(800, dw // 3))
            final_height = max(240, min(800, dh // 3))

            if self.debug:
                print(f"Bridge quilt: {qw}x{qh} ({cols}×{rows})", file=sys.stderr)
                print(f"Display size: {dw}x{dh}", file=sys.stderr)
                print(f"Window size: {final_width}x{final_height}", file=sys.stderr)

            self.window.set_size(final_width, final_height)
            self.window.center_on_screen()

            self._init_quilt_buffers()
            self.bridge_ok = True
            print(f"Bridge ready: quilt {qw}x{qh} ({cols}×{rows})", file=sys.stderr)
        except Exception as e:
            print("Bridge disabled:", e, file=sys.stderr)
            self.br_aspect = width / height

        # Viewport & default perspective
        GL.glViewport(0, 0, final_width, final_height)
        self.aspect = final_width / final_height
        self.camera.aspect = self.aspect
        self.proj_static = self._perspective(math.radians(fov), self.aspect, near, far)

        self._objects: List[Dict[str, Any]] = []
        self._dbg_ptr: Optional[ctypes._CFuncPtr] = None

        if self.debug:
            print("Controls:", file=sys.stderr)
            print("  ↑/↓ arrows: Focus (Shift×10)", file=sys.stderr)
            print("  ←/→ arrows: Depthiness (Shift×10)", file=sys.stderr)
            print("  ESC: Exit", file=sys.stderr)
            print(f"Initial - Focus: {self.focus:.2f}, Depthiness: {self.offset:.2f}", file=sys.stderr)

    # ------------ input helpers ------------
    def _handle_keyboard_input(self, dt: float) -> None:
        import glfw

        try:
            fast = 10.0 if (
                self.window.is_key_pressed(glfw.KEY_LEFT_SHIFT)
                or self.window.is_key_pressed(glfw.KEY_RIGHT_SHIFT)
            ) else 1.0
            step = self._input_adjustment_speed * dt * fast

            if self.window.is_key_pressed(glfw.KEY_UP):
                self.adjust_focus(step)
            if self.window.is_key_pressed(glfw.KEY_DOWN):
                self.adjust_focus(-step)
            if self.window.is_key_pressed(glfw.KEY_LEFT):
                self.adjust_offset(-step)
            if self.window.is_key_pressed(glfw.KEY_RIGHT):
                self.adjust_offset(step)
            if self.window.is_key_pressed(glfw.KEY_ESCAPE):
                glfw.set_window_should_close(self.window.handle, True)
        except Exception as e:
            if self.debug:
                print(f"Keyboard input failed: {e}", file=sys.stderr)

    def _handle_mouse_input(self, _: float) -> None:
        pass  # future use

    def _process_input(self, dt: float) -> None:
        self._handle_keyboard_input(dt)
        self._handle_mouse_input(dt)

    # ------------ public object API ------------
    def add_object(self, mesh: Mesh, shader: Shader, model: Optional[np.ndarray] = None) -> int:
        if model is None:
            model = np.eye(4, dtype=np.float32)
        loc = GL.glGetUniformLocation(shader.id, "u_mvp")
        self._objects.append({"mesh": mesh, "shader": shader, "loc": loc, "model": model})
        return len(self._objects) - 1

    def update_model(self, idx: int, model: np.ndarray) -> None:
        self._objects[idx]["model"] = model

    # ------------ focus / offset ------------
    def update_focus(self, focus: float) -> None:
        self.focus = max(-10.0, min(10.0, focus))
        print(f"Focus: {self.focus:.2f}", file=sys.stderr)

    def update_offset(self, offset: float) -> None:
        self.offset = max(0.0, min(5.0, offset))
        print(f"Depthiness: {self.offset:.2f}", file=sys.stderr)

    def adjust_focus(self, delta: float) -> None:
        self.update_focus(self.focus + delta)

    def adjust_offset(self, delta: float) -> None:
        self.update_offset(self.offset + delta)

    # ------------ rendering ------------
    def render_frame(self, dt: float = 0.016) -> None:
        self._process_input(dt)

        # ----- primary window: use center view (normalized 0.5)
        center_view, center_proj = self.camera.compute_view_projection_matrices(
            0.5, self.lkg_invert, self.offset, self.focus
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        fbw, fbh = self.window.framebuffer_size()
        GL.glViewport(0, 0, fbw, fbh)
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self._draw_objects(center_view, center_proj)

        # ----- quilt views
        if self.bridge_ok:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.quilt_fbo)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            vw, vh = self.qw // self.cols, self.qh // self.rows
            total = self.cols * self.rows
            for y in range(self.rows):
                for x in range(self.cols):
                    idx = y * self.cols + x
                    nrm = idx / (total - 1) if total > 1 else 0.5
                    GL.glViewport(x * vw, (self.rows - 1 - y) * vh, vw, vh)
                    v_mat, p_mat = self.camera.compute_view_projection_matrices(
                        nrm, self.lkg_invert, self.offset, self.focus
                    )
                    self._draw_objects(v_mat, p_mat)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            self.bridge.draw_interop_quilt_texture_gl(
                self.br_wnd,
                self.quilt_tex,
                PixelFormats.RGBA,
                self.qw,
                self.qh,
                self.cols,
                self.rows,
                self.br_aspect,
                1.0,
            )

        if self.debug:
            self._check_error("frame")
        self.window.swap_buffers()
        self.window.poll_events()

    # ------------ lifecycle ------------
    def should_close(self) -> bool:
        return self.window.should_close()

    def close(self) -> None:
        if self.bridge_ok:
            try:
                GL.glDeleteFramebuffers(1, [self.quilt_fbo])
                GL.glDeleteTextures(1, [self.quilt_tex])
                GL.glDeleteRenderbuffers(1, [self.depth_rb])
            except Exception:
                pass
        self.window.terminate()

    # ------------ internal helpers ------------
    def _draw_objects(self, view: np.ndarray, proj: np.ndarray) -> None:
        for obj in self._objects:
            mvp = proj @ view @ obj["model"]
            obj["shader"].use()
            GL.glUniformMatrix4fv(obj["loc"], 1, GL.GL_TRUE, mvp)
            obj["mesh"].draw(GL.GL_TRIANGLES)

    def _init_quilt_buffers(self) -> None:
        self.quilt_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.quilt_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.qw, self.qh, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None
        )
        self.depth_rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth_rb)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, self.qw, self.qh)
        self.quilt_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.quilt_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.quilt_tex, 0
        )
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.depth_rb
        )
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Quilt FBO incomplete")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    @staticmethod
    def _perspective(fov: float, aspect: float, n: float, f: float) -> np.ndarray:
        t = 1.0 / math.tan(fov / 2.0)
        return np.array(
            [
                [t / aspect, 0, 0, 0],
                [0, t, 0, 0],
                [0, 0, (f + n) / (n - f), (2 * f * n) / (n - f)],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # ------------ debug helpers ------------
    def _enable_khr_debug(self) -> None:
        def _cb(src, typ, _id, sev, length, msg, _):
            print("GL:", ctypes.string_at(msg, length).decode(), file=sys.stderr)

        try:
            self._dbg_ptr = GLDEBUGPROC_Core(_cb)
            GL.glEnable(GL.GL_DEBUG_OUTPUT)
            GL.glEnable(GL.GL_DEBUG_OUTPUT_SYNCHRONOUS)
            glDbgCB_Core(self._dbg_ptr, None)
            return
        except Exception:
            pass
        try:
            self._dbg_ptr = GLDEBUGPROC_ARB(_cb)
            GL.glEnable(GL.GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB)
            glDbgCB_ARB(self._dbg_ptr, None)
        except Exception as e:
            print("KHR_debug unavailable:", e, file=sys.stderr)
            self._dbg_ptr = None

    @staticmethod
    def _check_error(tag: str) -> None:
        err = GL.glGetError()
        if err:
            print(f"GL ERROR {tag}: 0x{err:04X}", file=sys.stderr)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
