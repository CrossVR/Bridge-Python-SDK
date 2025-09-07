#!/usr/bin/env python3
# DisplayQuilt.py
import sys
import os
import argparse
import re
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display a Looking Glass quilt from an image, URL, or video file."
    )
    parser.add_argument(
        "quilt",
        help="Path or URL to quilt image or video file (e.g., PNG, JPEG, MP4)."
    )
    parser.add_argument(
        "--cols",
        type=int,
        help="Number of columns in the quilt (override).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        help="Number of rows in the quilt (override).",
    )
    parser.add_argument(
        "--aspect",
        type=float,
        help="Display aspect ratio override. If omitted, parsed from filename or computed.",
    )
    args = parser.parse_args()

    base = os.path.basename(args.quilt)
    match = re.search(r"qs(\d+)x(\d+)a(\d+(?:\.\d+)?|\.\d+)", base)
    parsed_cols = int(match.group(1)) if match else None
    parsed_rows = int(match.group(2)) if match else None
    parsed_aspect = float(match.group(3)) if match else None

    ext = os.path.splitext(base)[1].lower()
    is_video = ext in (".mp4", ".mov", ".avi", ".mkv", ".webm")

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
                args.quilt
            ])
            w_vid, h_vid = map(int, probe.decode().strip().split(","))
        except Exception as e:
            print(f"Failed to probe video: {e}", file=sys.stderr)
            sys.exit(1)
        w, h = w_vid, h_vid
    else:
        try:
            if args.quilt.startswith(("http://", "https://")):
                with urllib.request.urlopen(args.quilt) as resp:
                    data_bytes = resp.read()
                image = Image.open(io.BytesIO(data_bytes)).convert("RGBA")
            else:
                image = Image.open(args.quilt).convert("RGBA")
        except Exception as e:
            print(f"Failed to load image: {e}", file=sys.stderr)
            sys.exit(1)
        data = np.array(image, dtype=np.uint8)
        h, w, _ = data.shape

    if args.cols is None and args.rows is None:
        if parsed_cols is not None and parsed_rows is not None:
            cols = parsed_cols
            rows = parsed_rows
        else:
            print(
                "Error: --cols and --rows must both be specified or encoded in filename as 'qs<cols>x<rows>a<aspect>'.",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.cols is not None and args.rows is not None:
        cols = args.cols
        rows = args.rows
    else:
        print(
            "Error: Provide both --cols and --rows, or neither to parse from filename.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.aspect is not None:
        aspect = args.aspect
    elif parsed_aspect is not None:
        aspect = parsed_aspect
    else:
        aspect = (w / cols) / (h / rows)

    # Initialize GLFW and create a hidden dummy window for a GL context
    if not glfw.init():
        print("Error: failed to initialize GLFW", file=sys.stderr)
        sys.exit(1)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    dummy = glfw.create_window(1, 1, "", None, None)
    if not dummy:
        print("Error: failed to create hidden GLFW window", file=sys.stderr)
        glfw.terminate()
        sys.exit(1)
    glfw.make_context_current(dummy)

    bridge = BridgeAPI()
    if not bridge.initialize("DisplayQuilt"):
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
    # We keep the dummy context current for all GL calls

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
            "-i", args.quilt,
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "pipe:1"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_size = w * h * 4

    # Main loop on the hidden window
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

        bridge.draw_interop_quilt_texture_gl(
            br_wnd,
            tex,
            PixelFormats.RGBA,
            w,
            h,
            cols,
            rows,
            aspect,
            1.0,
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
