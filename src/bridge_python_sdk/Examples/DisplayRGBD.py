#!/usr/bin/env python3
# DisplayRGBD.py
import sys
import os
import argparse
import io
import urllib.request
import subprocess
import shutil
import glfw

from PIL import Image
import numpy as np
from OpenGL import GL

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bridge_python_sdk.BridgeApi import BridgeAPI, PixelFormats

def parse_args():
    parser = argparse.ArgumentParser(
        description="Display a Looking Glass RGBD image or video from a file or URL."
    )
    parser.add_argument(
        "rgbd",
        help="Path or URL to RGBD image or video file (RGBA channels, with A=depth)."
    )
    parser.add_argument(
        "--depth-loc",
        type=int,
        default=2,
        help="Channel index (0–3) where depth is stored. 0 = top, 1 = bottom, 2 = left, 3 = right (default: 2)."
    )
    parser.add_argument(
        "--depthiness",
        type=float,
        default=1.0,
        help="Depthiness multiplier (range: 0 to 3, default: 1.0)."
    )
    parser.add_argument(
        "--focus",
        type=float,
        default=0.0,
        help="Focus value (range: -1 to 1, default: 0.0)."
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    base = os.path.basename(args.rgbd)
    ext = os.path.splitext(base)[1].lower()
    is_video = ext in (".mp4", ".mov", ".avi", ".mkv", ".webm")
    is_local = True
    mtime = 0.0

    # Load dims (and data for images)
    if is_video:
        if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
            print(
                "Error: ffmpeg and ffprobe must be installed and on your PATH to play video.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            probe = subprocess.check_output([
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                args.rgbd
            ])
            w, h = map(int, probe.decode().strip().split(","))
        except Exception as e:
            print(f"Failed to probe video: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            if args.rgbd.startswith(("http://", "https://")):
                is_local = False
                with urllib.request.urlopen(args.rgbd) as resp:
                    buf = resp.read()
                image = Image.open(io.BytesIO(buf)).convert("RGBA")
            else:
                image = Image.open(args.rgbd).convert("RGBA")
        except Exception as e:
            print(f"Failed to load RGBD image: {e}", file=sys.stderr)
            sys.exit(1)
        data = np.array(image, dtype=np.uint8)
        h, w, _ = data.shape

    if is_local:
        mtime = os.path.getmtime(args.rgbd)

    gl_major = 4
    gl_minor = 3
    core_profile = True

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
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    
    dummy = glfw.create_window(1, 1, "", None, None)
    if not dummy:
        print("Error: failed to create hidden GLFW window", file=sys.stderr)
        glfw.terminate()
        sys.exit(1)
    glfw.make_context_current(dummy)

    bridge = BridgeAPI()
    if not bridge.initialize("DisplayRGBD"):
        print("Bridge initialize failed", file=sys.stderr)
        glfw.destroy_window(dummy)
        glfw.terminate()
        sys.exit(1)

    br_wnd = bridge.instance_window_gl(-1)
    if br_wnd == 0:
        print("Bridge.instance_window_gl returned 0", file=sys.stderr)
        glfw.destroy_window(dummy)
        glfw.terminate()
        sys.exit(1)

    # Query display's default quilt settings
    asp, qw, qh, cols, rows = bridge.get_default_quilt_settings(br_wnd)
    aspect = float(asp)

    # Create GL texture
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGBA8,
        w,
        h,
        0,
        GL.GL_RGBA,
        GL.GL_UNSIGNED_BYTE,
        None if is_video else data,
    )

    if is_video:
        proc = subprocess.Popen([
            "ffmpeg",
            "-i", args.rgbd,
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "pipe:1"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_size = w * h * 4

    # Normalize focus exactly as in REST API
    focus_min = 0.005
    focus_max = -0.007
    bridge_focus = args.focus * args.depthiness
    normalized_focus = focus_min + ((bridge_focus + 1.0) / 2.0) * (focus_max - focus_min)

    # Main display loop
    while not glfw.window_should_close(dummy):
        if is_video:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D,
                0,
                0,
                0,
                w,
                h,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                frame,
            )
        elif is_local and mtime != os.path.getmtime(args.rgbd):
            try:
                image = Image.open(args.rgbd).convert("RGBA")
                data = np.array(image, dtype=np.uint8)
                h, w, _ = data.shape
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_RGBA8,
                    w,
                    h,
                    0,
                    GL.GL_RGBA,
                    GL.GL_UNSIGNED_BYTE,
                    data,
                )
                mtime = os.path.getmtime(args.rgbd)
            except Exception as e:
                print(f"Failed to load RGBD image: {e}", file=sys.stderr)

        bridge.draw_interop_rgbd_texture_gl(
            br_wnd,
            tex,
            PixelFormats.RGBA,
            w,               # input width
            h,               # input height
            qw,              # quilt width
            qh,              # quilt height
            cols,            # quilt columns
            rows,            # quilt rows
            aspect,          # display aspect
            normalized_focus,# normalized focus
            args.depthiness, # depthiness
            1.0,             # zoom (unused)
            args.depth_loc   # depth channel index
        )
        glfw.swap_buffers(dummy)
        glfw.poll_events()

    if is_video:
        proc.terminate()
    GL.glDeleteTextures(1, [tex])
    glfw.destroy_window(dummy)
    glfw.terminate()

if __name__ == "__main__":
    main()
