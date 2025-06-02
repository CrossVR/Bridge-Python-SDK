#!/usr/bin/env python3
# Render.py – quilt-aware renderer for Bridge SDK
import math, sys, ctypes
from typing import Optional, Dict, Any, List

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
                 fov:float=60.0, near:float=0.1, far:float=100.0,
                 debug:bool=False):
        self.debug=debug
        self.window=Window(width,height,title)
        if self.debug:
            self._enable_khr_debug()
            print("GL_VERSION:",GL.glGetString(GL.GL_VERSION).decode(),file=sys.stderr)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK,GL.GL_LINE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glViewport(0,0,width,height)
        aspect=width/height
        self.proj=self._perspective(math.radians(fov),aspect,near,far)
        self.view=np.eye(4,dtype=np.float32); self.view[2,3]=-3.0
        self._objects:List[Dict[str,Any]]=[]
        self._dbg_ptr:Optional[ctypes._CFuncPtr]=None
        # ---- Bridge setup
        self.bridge_ok=False
        try:
            self.bridge=BridgeAPI()
            if not self.bridge.initialize("BridgePythonSample"):
                raise RuntimeError("Bridge initialize failed")
            self.br_wnd=self.bridge.instance_window_gl(-1)
            asp,qw,qh,cols,rows=self.bridge.get_default_quilt_settings(self.br_wnd)
            self.br_aspect=float(asp); self.qw=qw; self.qh=qh; self.cols=cols; self.rows=rows
            self._init_quilt_buffers()
            self.bridge_ok=True
            print("Bridge ready: quilt {}x{} ({}×{})".format(self.qw,self.qh,self.cols,self.rows),file=sys.stderr)
        except Exception as e:
            print("Bridge disabled:",e,file=sys.stderr)
            self.bridge_ok=False

    # -------------------------------------------------------------- public API
    def add_object(self,mesh:Mesh,shader:Shader,model_matrix:Optional[np.ndarray]=None)->int:
        if model_matrix is None: model_matrix=np.eye(4,dtype=np.float32)
        loc=GL.glGetUniformLocation(shader.id,"u_mvp")
        self._objects.append({"mesh":mesh,"shader":shader,"loc":loc,"model":model_matrix})
        return len(self._objects)-1

    def update_model(self,h:int,model:np.ndarray)->None:
        self._objects[h]["model"]=model

    def render_frame(self)->None:
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
                    norm=idx/(total-1)
                    GL.glViewport(x*view_w,(self.rows-1-y)*view_h,view_w,view_h)
                    self._draw_objects(self._camera_view(norm))
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)
            self.bridge.draw_interop_quilt_texture_gl(
                self.br_wnd,self.quilt_tex,PixelFormats.RGBA,
                self.qw,self.qh,self.cols,self.rows,self.br_aspect,1.0)
        if self.debug: self._check_error("frame")
        self.window.swap_buffers(); self.window.poll_events()

    def should_close(self)->bool: return self.window.should_close()

    # ------------------------------------------------------------ internals
    def _draw_objects(self,view_mat:np.ndarray)->None:
        for obj in self._objects:
            mvp=self.proj@view_mat@obj["model"]
            obj["shader"].use()
            GL.glUniformMatrix4fv(obj["loc"],1,GL.GL_TRUE,mvp)
            obj["mesh"].draw(GL.GL_TRIANGLES)

    def _camera_view(self,t:float)->np.ndarray:
        """Stub: slide camera horizontally across [-1,1] based on t∈[0,1]."""
        v=self.view.copy()
        v[0,3]=(t*2.0-1.0)*0.1  # simple toe-in
        return v

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
