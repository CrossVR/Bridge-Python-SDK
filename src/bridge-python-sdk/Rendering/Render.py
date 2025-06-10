#!/usr/bin/env python3
# Render.py – quilt-aware renderer for Bridge SDK
import math, sys, ctypes
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL.raw.GL.VERSION.GL_4_3 import glDebugMessageCallback as glDbgCB_Core
from OpenGL.raw.GL.VERSION.GL_4_3 import GLDEBUGPROC              as GLDEBUGPROC_Core
from OpenGL.raw.GL.ARB.debug_output import glDebugMessageCallbackARB as glDbgCB_ARB
from OpenGL.raw.GL.ARB.debug_output import GLDEBUGPROCARB            as GLDEBUGPROC_ARB

from BridgeApi import BridgeAPI, PixelFormats
from .Window     import Window
from .Shader     import Shader
from .Mesh       import Mesh


class Render:
    def __init__(self,
                 width:int=800, height:int=600, title:str="",
                 fov:float=14.0, near:float=0.1, far:float=100.0,
                 debug:bool=False,
                 # LKG camera parameters
                 lkg_size:float=3.0,
                 lkg_center:Optional[np.ndarray]=None,
                 lkg_up:Optional[np.ndarray]=None,
                 lkg_viewcone:float=40.0,
                 lkg_invert:bool=False):
        self.debug=debug
        
        # Create primary window first with default size
        self.window=Window(width, height, title)
        
        if self.debug:
            self._enable_khr_debug()
            print("GL_VERSION:",GL.glGetString(GL.GL_VERSION).decode(),file=sys.stderr)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK,GL.GL_LINE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        
        # Camera parameters
        self.fov = fov
        self.near = near
        self.far = far
        
        # LKG camera parameters
        self.lkg_size = lkg_size
        self.lkg_center = lkg_center if lkg_center is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.lkg_up = lkg_up if lkg_up is not None else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.lkg_viewcone = lkg_viewcone
        self.lkg_invert = lkg_invert
        
        # Scene-level focus controls (matching C# Scene class)
        self.focus = 0.0        # Focus parameter (C# Scene.Focus)
        self.offset = 1.0       # Offset parameter (C# Scene.Offset, becomes depthiness)
        
        # Input handling
        self._input_adjustment_speed = 2.0
        
        # ---- Bridge setup after OpenGL context exists
        self.bridge_ok=False
        final_width, final_height = width, height
        try:
            self.bridge=BridgeAPI()
            if not self.bridge.initialize("BridgePythonSample"):
                raise RuntimeError("Bridge initialize failed")
            
            # Now that Bridge is initialized, try to create the window instance
            self.br_wnd=self.bridge.instance_window_gl(-1)
            if self.br_wnd == 0:
                raise RuntimeError("instance_window_gl returned 0")
                
            asp,qw,qh,cols,rows=self.bridge.get_default_quilt_settings(self.br_wnd)
            self.br_aspect=float(asp); self.qw=qw; self.qh=qh; self.cols=cols; self.rows=rows
            
            # Try to get actual display dimensions instead of quilt dimensions
            try:
                # Get calibration data which contains the actual screen dimensions
                cal_data = self.bridge.get_calibration_gl(self.br_wnd)
                if cal_data and hasattr(cal_data, 'screenW') and hasattr(cal_data, 'screenH'):
                    display_width = cal_data.screenW
                    display_height = cal_data.screenH
                    if self.debug:
                        print(f"Using actual display size: {display_width}x{display_height}", file=sys.stderr)
                else:
                    # Fallback: estimate display size from aspect ratio and reasonable dimensions
                    if abs(asp - (16/9)) < 0.1:  # 16:9 display
                        display_width, display_height = 1920, 1080
                    elif abs(asp - (9/16)) < 0.1:  # Portrait 9:16 display (like Looking Glass Portrait)
                        display_width, display_height = 1440, 2560
                    else:
                        # Use quilt dimensions but limit to reasonable size
                        display_width = min(qw, 2048)
                        display_height = min(qh, 2048)
                    if self.debug:
                        print(f"Estimated display size: {display_width}x{display_height}", file=sys.stderr)
                        
            except Exception as e:
                # Final fallback: reasonable default based on aspect ratio
                if asp > 1.0:  # Landscape
                    display_width, display_height = 1920, 1080
                else:  # Portrait
                    display_width, display_height = 1440, 2560
                if self.debug:
                    print(f"Using fallback display size: {display_width}x{display_height}", file=sys.stderr)
            
            # Calculate window size as 1/3 of actual display, with reasonable limits
            target_width = display_width // 3
            target_height = display_height // 3
            
            # Ensure minimum usable size and maximum reasonable size
            final_width = max(320, min(800, target_width))   # Between 320 and 800 pixels wide
            final_height = max(240, min(800, target_height)) # Between 240 and 800 pixels tall
            
            if self.debug:
                print(f"Bridge quilt: {qw}x{qh} ({cols}×{rows})", file=sys.stderr)
                print(f"Display size: {display_width}x{display_height}", file=sys.stderr)
                print(f"Window size: {final_width}x{final_height}", file=sys.stderr)
            
            # Resize and center the window
            self.window.set_size(final_width, final_height)
            self.window.center_on_screen()
            
            self._init_quilt_buffers()
            self.bridge_ok=True
            print("Bridge ready: quilt {}x{} ({}×{})".format(self.qw,self.qh,self.cols,self.rows),file=sys.stderr)
            
        except Exception as e:
            print("Bridge disabled:",e,file=sys.stderr)
            self.bridge_ok=False
            self.br_aspect = width / height
            # Use default size if Bridge fails
            final_width, final_height = width, height
        
        # Set up viewport and aspect ratio
        GL.glViewport(0, 0, final_width, final_height)
        aspect = final_width / final_height
        self.aspect = aspect
        
        self.proj=self._perspective(math.radians(fov),aspect,near,far)
        self.view=np.eye(4,dtype=np.float32); self.view[2,3]=-3.0
        self._objects:List[Dict[str,Any]]=[]
        self._dbg_ptr:Optional[ctypes._CFuncPtr]=None
        
        # Debug LKG parameters
        if self.debug:
            print("Controls:", file=sys.stderr)
            print("  ↑/↓ arrows: Adjust focus", file=sys.stderr)
            print("  ←/→ arrows: Adjust depthiness", file=sys.stderr)
            print("  ESC: Exit application", file=sys.stderr)
            print(f"LKG Camera - Size: {self.lkg_size}, Viewcone: {self.lkg_viewcone}, FOV: {self.fov}", file=sys.stderr)
            print(f"Window size: {final_width}x{final_height}, Aspect: {aspect:.3f}", file=sys.stderr)
            print(f"Initial - Focus: {self.focus:.2f}, Depthiness: {self.offset:.2f}", file=sys.stderr)

    # -------------------------------------------------------------- Input Handling
    def _handle_keyboard_input(self, delta_time: float) -> None:
        """Handle keyboard input for focus and depthiness control"""
        import glfw  # Import only when needed
        
        try:
            # Use Window's built-in keyboard methods
            up_pressed = self.window.is_key_pressed(glfw.KEY_UP)
            down_pressed = self.window.is_key_pressed(glfw.KEY_DOWN)
            left_pressed = self.window.is_key_pressed(glfw.KEY_LEFT)
            right_pressed = self.window.is_key_pressed(glfw.KEY_RIGHT)
            escape_pressed = self.window.is_key_pressed(glfw.KEY_ESCAPE)
            
            # Handle focus control (up/down arrows)
            if up_pressed:
                self.adjust_focus(self._input_adjustment_speed * delta_time)
            if down_pressed:
                self.adjust_focus(-self._input_adjustment_speed * delta_time)
            
            # Handle depthiness/offset control (left/right arrows)
            if left_pressed:
                self.adjust_offset(-self._input_adjustment_speed * delta_time)
            if right_pressed:
                self.adjust_offset(self._input_adjustment_speed * delta_time)
            
            # Handle exit
            if escape_pressed:
                import glfw
                glfw.set_window_should_close(self.window.handle, True)
                
        except Exception as e:
            if self.debug:
                print(f"Keyboard input failed: {e}", file=sys.stderr)

    def _handle_mouse_input(self, delta_time: float) -> None:
        """Handle mouse input for additional controls"""
        try:
            # Get mouse position and buttons
            mouse_x, mouse_y = self.window.get_mouse_pos()
            left_button = self.window.is_mouse_button_pressed(0)  # Left mouse button
            right_button = self.window.is_mouse_button_pressed(1)  # Right mouse button
            
            # Example: Use mouse wheel for focus adjustment (if scroll callback is set)
            # This could be extended for more mouse-based controls
            
        except Exception:
            pass  # Silently ignore mouse input errors

    def _process_input(self, delta_time: float) -> None:
        """Process all input for this frame"""
        self._handle_keyboard_input(delta_time)
        self._handle_mouse_input(delta_time)

    # -------------------------------------------------------------- public API
    def add_object(self,mesh:Mesh,shader:Shader,model_matrix:Optional[np.ndarray]=None)->int:
        if model_matrix is None: model_matrix=np.eye(4,dtype=np.float32)
        loc=GL.glGetUniformLocation(shader.id,"u_mvp")
        self._objects.append({"mesh":mesh,"shader":shader,"loc":loc,"model":model_matrix})
        return len(self._objects)-1

    def update_model(self,h:int,model:np.ndarray)->None:
        self._objects[h]["model"]=model

    def update_focus(self, focus: float) -> None:
        """Update focus parameter (clamped to -4.0 to 4.0 like C# version)"""
        old_focus = self.focus
        self.focus = max(-10.0, min(10.0, focus))
        if self.debug and abs(old_focus - self.focus) > 0.01:
            print(f"Focus: {self.focus:.2f}", file=sys.stderr)

    def update_offset(self, offset: float) -> None:
        """Update offset parameter (clamped to 0.0 to 2.0 like C# version)"""
        old_offset = self.offset
        self.offset = max(0.0, min(2.0, offset))
        if self.debug and abs(old_offset - self.offset) > 0.01:
            print(f"Depthiness: {self.offset:.2f}", file=sys.stderr)

    def adjust_focus(self, delta: float) -> None:
        """Adjust focus by delta amount"""
        self.update_focus(self.focus + delta)

    def adjust_offset(self, delta: float) -> None:
        """Adjust offset by delta amount"""
        self.update_offset(self.offset + delta)

    def set_input_speed(self, speed: float) -> None:
        """Set the adjustment speed for keyboard input"""
        self._input_adjustment_speed = speed

    def set_window_title(self, title: str) -> None:
        """Update window title (useful for showing current values)"""
        self.window.set_title(title)

    def center_window(self) -> None:
        """Center the window on screen"""
        self.window.center_on_screen()

    def render_frame(self, delta_time: float = 0.016) -> None:
        """Render a frame with input processing"""
        # Process input first
        self._process_input(delta_time)
        
        # ----- primary window
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)
        fbw,fbh=self.window.framebuffer_size()
        GL.glViewport(0,0,fbw,fbh)
        GL.glClearColor(0.1,0.1,0.1,1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)
        self._draw_objects(self.view)
        
        # ----- quilt / Looking Glass
        if self.bridge_ok:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,self.quilt_fbo)
            GL.glClearColor(0.0,0.0,0.0,1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT)
            view_w=self.qw//self.cols; view_h=self.qh//self.rows
            total=self.cols*self.rows
            for y in range(self.rows):
                for x in range(self.cols):
                    idx=y*self.cols+x
                    norm=idx/(total-1) if total > 1 else 0.5
                    GL.glViewport(x*view_w,(self.rows-1-y)*view_h,view_w,view_h)
                    lkg_view, lkg_proj = self._compute_lkg_view_projection_matrices(
                        norm, self.lkg_invert, self.offset, self.focus)
                    self._draw_objects(lkg_view, lkg_proj)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)
            self.bridge.draw_interop_quilt_texture_gl(
                self.br_wnd,self.quilt_tex,PixelFormats.RGBA,
                self.qw,self.qh,self.cols,self.rows,self.br_aspect,1.0)
        
        if self.debug: self._check_error("frame")
        self.window.swap_buffers()
        self.window.poll_events()

    def should_close(self)->bool: 
        return self.window.should_close()

    def close(self) -> None:
        """Properly close the renderer and clean up resources"""
        if self.bridge_ok:
            try:
                # Clean up Bridge resources
                GL.glDeleteFramebuffers(1, [self.quilt_fbo])
                GL.glDeleteTextures(1, [self.quilt_tex])
                GL.glDeleteRenderbuffers(1, [self.depth_rb])
            except:
                pass
        self.window.terminate()

    # ------------------------------------------------------------ LKG Camera Implementation
    def _get_camera_distance(self) -> float:
        """Get the camera's distance from center of focal plane, given FOV - matches C# GetCameraDistance exactly"""
        # C# code: return Size / MathF.Tan(Fov * (MathF.PI / 180.0f));
        # This uses FULL FOV, not half FOV!
        fov_rad = math.radians(self.fov)
        return self.lkg_size / math.tan(fov_rad)

    def _get_camera_offset(self) -> float:
        """Get the camera's offset based on the viewcone - matches C# GetCameraOffset exactly"""
        # C# code: return GetCameraDistance() * MathF.Tan(Viewcone * (MathF.PI / 180.0f));
        # This also uses FULL viewcone, not half!
        viewcone_rad = math.radians(self.lkg_viewcone)
        return self._get_camera_distance() * math.tan(viewcone_rad)

    def _compute_lkg_view_matrix(self, offset: float) -> np.ndarray:
        """Helper method to compute the view matrix - matches C# ComputeViewMatrix"""
        # Compute forward vector f (looking down positive Z)
        f = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        # Compute up vector u = normalize(up)
        u = self.lkg_up / np.linalg.norm(self.lkg_up)
        
        # Compute s = normalize(cross(f, u))
        s_cross = np.cross(f, u)
        s = s_cross / np.linalg.norm(s_cross)
        
        # Recompute up vector u = cross(s, f)
        u = np.cross(s, f)
        
        # Build the view matrix in column-major order (OpenGL format)
        matrix = np.eye(4, dtype=np.float32)
        
        # Set rotation part (first 3x3)
        matrix[0, 0] = s[0]   # M11
        matrix[1, 0] = s[1]   # M21  
        matrix[2, 0] = s[2]   # M31
        
        matrix[0, 1] = u[0]   # M12
        matrix[1, 1] = u[1]   # M22
        matrix[2, 1] = u[2]   # M32
        
        matrix[0, 2] = -f[0]  # M13
        matrix[1, 2] = -f[1]  # M23
        matrix[2, 2] = -f[2]  # M33
        
        # Set translation part (4th column)
        matrix[0, 3] = offset                           # M41
        matrix[1, 3] = 0.0                             # M42
        matrix[2, 3] = -self._get_camera_distance()    # M43
        matrix[3, 3] = 1.0                             # M44
        
        return matrix

    def _compute_lkg_projection_matrix(self) -> np.ndarray:
        """Helper method to compute the projection matrix - matches C# ComputeProjectionMatrix"""
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        n = self.near
        f_p = self.far
        
        # Build projection matrix in column-major order
        matrix = np.zeros((4, 4), dtype=np.float32)
        
        matrix[0, 0] = f / self.aspect  # M11
        matrix[1, 1] = f                # M22
        matrix[2, 2] = (f_p + n) / (n - f_p)    # M33
        matrix[3, 2] = -1.0             # M43
        matrix[2, 3] = (2 * f_p * n) / (n - f_p)  # M34
        
        return matrix

    def _compute_lkg_view_projection_matrices(self, normalized_view: float, invert: bool, depthiness: float, focus: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute view and projection matrices for hologram views - matches C# ComputeViewProjectionMatrices"""
        # Adjust camera position based on normalized_view and depthiness
        offset = -(normalized_view - 0.5) * depthiness * self._get_camera_offset()
        
        # Adjust up vector if invert is true
        adjusted_up = np.array([self.lkg_up[0], -self.lkg_up[1], self.lkg_up[2]], dtype=np.float32) if invert else self.lkg_up.copy()
        
        # Temporarily set the up vector for view matrix computation
        original_up = self.lkg_up.copy()
        self.lkg_up = adjusted_up
        
        # Compute the view matrix with the adjusted position and up vector
        view_matrix = self._compute_lkg_view_matrix(offset)
        
        # Restore original up vector
        self.lkg_up = original_up
        
        # Apply X-flip if not inverted
        if not invert:
            flip_x = np.array([
                [-1.0,  0.0,  0.0,  0.0],
                [ 0.0,  1.0,  0.0,  0.0],
                [ 0.0,  0.0,  1.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0]
            ], dtype=np.float32)
            view_matrix = flip_x @ view_matrix
        
        # Compute the standard projection matrix
        projection_matrix = self._compute_lkg_projection_matrix()
        
        # Apply frustum shift to the projection matrix
        view_position = normalized_view
        center_position = 0.5
        distance_from_center = view_position - center_position
        frustum_shift = distance_from_center * focus
        
        # Modify the projection matrix to include frustum shift
        # M31 in C# corresponds to [2, 0] in column-major numpy array
        projection_matrix[2, 0] += (offset * 2.0 / (self.lkg_size * self.aspect)) + frustum_shift
        
        return view_matrix, projection_matrix

    def _camera_view(self, t: float) -> np.ndarray:
        """Backward compatibility: compute LKG camera view matrix for normalized view position t ∈ [0,1]."""
        view_matrix, _ = self._compute_lkg_view_projection_matrices(t, self.lkg_invert, self.offset, self.focus)
        return view_matrix

    # ------------------------------------------------------------ internals
    def _draw_objects(self, view_mat: np.ndarray, proj_mat: Optional[np.ndarray] = None) -> None:
        if proj_mat is None:
            proj_mat = self.proj
            
        for obj in self._objects:
            mvp = proj_mat @ view_mat @ obj["model"]
            obj["shader"].use()
            GL.glUniformMatrix4fv(obj["loc"], 1, GL.GL_TRUE, mvp)
            obj["mesh"].draw(GL.GL_TRIANGLES)

    def _init_quilt_buffers(self)->None:
        self.quilt_tex=GL.glGenTextures(1); GL.glBindTexture(GL.GL_TEXTURE_2D,self.quilt_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER,GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER,GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D,0,GL.GL_RGBA8,self.qw,self.qh,0,GL.GL_RGBA,GL.GL_UNSIGNED_BYTE,None)
        self.depth_rb=GL.glGenRenderbuffers(1); GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,self.depth_rb)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER,GL.GL_DEPTH_COMPONENT24,self.qw,self.qh)
        self.quilt_fbo=GL.glGenFramebuffers(1); GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,self.quilt_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,GL.GL_COLOR_ATTACHMENT0,GL.GL_TEXTURE_2D,self.quilt_tex,0)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,GL.GL_DEPTH_ATTACHMENT,GL.GL_RENDERBUFFER,self.depth_rb)
        status=GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status!=GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Quilt FBO incomplete")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)

    @staticmethod
    def _perspective(fov:float,aspect:float,near:float,far:float)->np.ndarray:
        f=1.0/math.tan(fov*0.5)
        return np.array([[f/aspect,0,0,0],
                         [0,f,0,0],
                         [0,0,(far+near)/(near-far),(2*far*near)/(near-far)],
                         [0,0,-1,0]],dtype=np.float32)

    # --------------------------------------------------------- debug helpers
    def _enable_khr_debug(self)->None:
        def _cb(src,typ,_id,sev,len_,msg,_):
            print("GL:",ctypes.string_at(msg,len_).decode(),file=sys.stderr)
        try:
            self._dbg_ptr=GLDEBUGPROC_Core(_cb); GL.glEnable(GL.GL_DEBUG_OUTPUT); GL.glEnable(GL.GL_DEBUG_OUTPUT_SYNCHRONOUS); glDbgCB_Core(self._dbg_ptr,None); return
        except Exception: pass
        try:
            self._dbg_ptr=GLDEBUGPROC_ARB(_cb); GL.glEnable(GL.GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB); glDbgCB_ARB(self._dbg_ptr,None)
        except Exception as e:
            print("KHR_debug unavailable:",e,file=sys.stderr); self._dbg_ptr=None

    @staticmethod
    def _check_error(tag:str)->None:
        err=GL.glGetError()
        if err:
            print(f"GL ERROR {tag}: 0x{err:04X}",file=sys.stderr)

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.close()
        except:
            pass
