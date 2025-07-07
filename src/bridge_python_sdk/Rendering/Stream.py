#!/usr/bin/env python3
# Rendering/Stream.py — RGB-D UDP sender / receiver with robust A/V handling.
# Entire module is self-contained; drop-in replacement for the old file.

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import time
from typing import List, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import glfw
import numpy as np
from OpenGL import GL

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from BridgeApi import BridgeAPI, PixelFormats                                              # noqa: E402


# ════════════════════════════════════════════════════════════════════
#                           ffmpeg helpers
# ════════════════════════════════════════════════════════════════════
class _FFmpegMixin:
    _MIN_DIM    = 32
    _FIFO_BYTES = 5_000_000
    _RE_WXH     = re.compile(r"(\d+)[x,](\d+)")
    _RE_DEVICE  = re.compile(r"]\s*\"([^\"]+)\"\s+\(video\)", re.IGNORECASE)

    # ── spawn + log ───────────────────────────────────────────────────
    @staticmethod
    def _spawn(cmd: Sequence[str], tag: str, *, capture_out: bool = False) -> subprocess.Popen:
        logging.info("%s: exec %s", tag, " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if capture_out else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        threading.Thread(target=_FFmpegMixin._forward_log,
                         args=(proc.stderr, tag), daemon=True).start()
        return proc

    @staticmethod
    def _forward_log(stream, tag) -> None:
        for raw in iter(stream.readline, b""):
            if not raw:
                break
            logging.warning("%s | %s", tag, raw.decode(errors="replace").rstrip())

    # ── UDP helper ────────────────────────────────────────────────────
    @classmethod
    def _udp_url(cls, url: str) -> str:
        p = urlparse(url)
        if p.scheme not in ("udp", "prompeg+udp"):
            return url
        q = dict(parse_qsl(p.query))
        q.setdefault("fifo_size", str(cls._FIFO_BYTES))
        q.setdefault("overrun_nonfatal", "1")
        return urlunparse(p._replace(query=urlencode(q)))

    # ── dshow device list (numeric ↔ name) ────────────────────────────
    @staticmethod
    def _list_dshow_video_devices() -> List[str]:
        cmd = ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        devices, in_video = [], False
        for ln in proc.stderr.splitlines():
            if "DirectShow video devices" in ln:
                in_video = True
                continue
            if "DirectShow audio devices" in ln:
                break
            if "(video)" in ln.lower():
                m = _FFmpegMixin._RE_DEVICE.search(ln)
                if m:
                    devices.append(m.group(1))
        return devices

    # ── generic width×height probe ────────────────────────────────────
    @classmethod
    def _probe_ffprobe(cls, cmd: Sequence[str],
                       timeout: float | None = 10) -> Optional[tuple[int, int]]:
        logging.info("using ffprobe to get stream size")
        logging.debug(cmd)

        # build a probe-specific command with much smaller fifo and a demuxer timeout
        cmd2: list[str] = []
        for token in cmd:
            if token.startswith("udp://"):
                p = urlparse(token)
                if p.scheme in ("udp", "prompeg+udp"):
                    q = dict(parse_qsl(p.query))
                    q["fifo_size"]        = str(100_000)
                    q.setdefault("overrun_nonfatal", "1")
                    q.setdefault("timeout", "500000")
                    token = urlunparse(p._replace(query=urlencode(q)))
            cmd2.append(token)

        # also limit how much ffprobe analyzes before giving up
        if cmd2 and cmd2[0] == "ffprobe":
            cmd2.insert(1, "-probesize"); cmd2.insert(2, "32M")
            cmd2.insert(3, "-analyzeduration"); cmd2.insert(4, "1M")

        proc = subprocess.Popen(
            cmd2,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            logging.info("failed (timeout)")
            proc.kill()
            return None

        if proc.returncode != 0:
            logging.info("failed")
            return None

        for ln in out.splitlines():
            m = cls._RE_WXH.search(ln)
            if m:
                w, h = int(m[1]), int(m[2])
                if not w & 1 and w >= cls._MIN_DIM and h >= cls._MIN_DIM:
                    logging.info("succeeded")
                    return w, h

        return None


# ════════════════════════════════════════════════════════════════════
#                     shared OpenGL preview
# ════════════════════════════════════════════════════════════════════
def _norm_focus(f: float, d: float) -> float:
    a, b = 0.005, -0.007
    return a + (((f * d) + 1) / 2) * (b - a)


def _check_gl_error(label: str = "") -> None:
    err = GL.glGetError()
    if err != GL.GL_NO_ERROR:
        msg = GLU.gluErrorString(err)
        text = msg.decode() if msg not in (None, b'') else f"0x{err:04X}"
        raise RuntimeError(f"OpenGL error {text} at {label}")


def preview_loop(proc: subprocess.Popen,
                 w: int,
                 h: int,
                 depth_loc: int,
                 depth_scale: float,
                 focus: float,
                 diag: bool) -> None:

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
    win = glfw.create_window(1, 1, "", None, None)
    glfw.make_context_current(win)
    _check_gl_error("make_context_current")

    bridge = BridgeAPI()
    bridge.initialize("RGBD")
    handle = bridge.instance_window_gl(-1)
    asp, qw, qh, tx, ty = bridge.get_default_quilt_settings(handle)

    tex = GL.glGenTextures(1)
    _check_gl_error("glGenTextures")

    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    _check_gl_error("glBindTexture")

    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    _check_gl_error("glPixelStorei")

    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    _check_gl_error("glTexParameteri MIN_FILTER")

    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    _check_gl_error("glTexParameteri MAG_FILTER")

    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0,
                    GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    _check_gl_error("glTexImage2D")

    frame_bytes = w * h * 4
    buf = bytearray()
    frames = 0
    t0 = time.time()
    f_norm = _norm_focus(focus, depth_scale)

    while not glfw.window_should_close(win):
        while len(buf) < frame_bytes and proc.poll() is None:
            chunk = proc.stdout.read(frame_bytes - len(buf))
            if not chunk:
                time.sleep(0.002)
                continue
            buf.extend(chunk)
        if len(buf) < frame_bytes:
            break

        raw = bytes(buf[:frame_bytes])
        del buf[:frame_bytes]

        try:
            rgba = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
        except ValueError:
            buf.clear()
            continue

        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        _check_gl_error("glBindTexture (frame upload)")

        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                           GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba)
        _check_gl_error("glTexSubImage2D")

        bridge.draw_interop_rgbd_texture_gl(
            handle, tex, PixelFormats.RGBA,
            w, h, qw, qh, tx, ty,
            float(asp), f_norm, depth_scale, 1.0, depth_loc,
        )
        _check_gl_error("bridge.draw_interop_rgbd_texture_gl")

        glfw.swap_buffers(win)
        glfw.poll_events()
        _check_gl_error("swap_buffers")

        frames += 1
        if time.time() - t0 >= 1:
            logging.info("FPS %d", frames)
            if diag:
                logging.debug("RGBA[0,0]=%s", tuple(rgba[0, 0]))
            frames = 0
            t0 = time.time()

    GL.glDeleteTextures(1, [tex])
    _check_gl_error("glDeleteTextures")

    glfw.destroy_window(win)
    glfw.terminate()
    
# ════════════════════════════════════════════════════════════════════
#                         StreamReceiver
# ════════════════════════════════════════════════════════════════════
class StreamReceiver(_FFmpegMixin):
    """Receives an MPEG-TS / ProMPEG UDP stream and previews video **with synchronous audio**."""

    _AUDIO_PORT = 54002  # localhost port for tee’d audio

    def __init__(self, args):
        logging.info("Stream Receiver created")
        self.a = args

    # ───── ffmpeg demux/tee: raw RGBA to stdout, audio to local UDP ───
    def _dec_cmd(self) -> list[str]:
        audio_url = f"udp://127.0.0.1:{self._AUDIO_PORT}"
        return [
            "ffmpeg", "-v", "warning", "-fflags", "nobuffer", "-flags", "low_delay",
            "-i", self._udp_url(self.a.url),
            "-map", "0:v:0", "-vf", "format=rgba",
            "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1",
            "-map", "0:a:0?", "-c:a", "copy",
            "-f", "mpegts", audio_url,
        ]

    # ───── tiny ffplay instance just for audio ────────────────────────
    def _audio_cmd(self) -> list[str]:
        return [
            "ffplay", "-v", "warning",
            "-nodisp", "-autoexit", "-fflags", "nobuffer",
            f"udp://127.0.0.1:{self._AUDIO_PORT}",
        ]

    # ───── public run() ───────────────────────────────────────────────
    def run(self) -> None:
        backoff = 1
        while True:
            dec: Optional[subprocess.Popen] = None
            audio: Optional[subprocess.Popen] = None
            try:
                # ── detect resolution ────────────────────────────────
                if self.a.width and self.a.height:
                    w, h = self.a.width, self.a.height
                else:
                    probe = self._probe_ffprobe(
                        [
                            "ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height",
                            "-of", "csv=s=x:p=0",
                            self.a.url,
                        ],
                        timeout=self.a.wait,
                    ) or (640, 360)
                    w, h = probe
                logging.info("stream resolution %dx%d", w, h)

                # ── start unified demux/tee ─────────────────────────
                dec = self._spawn(self._dec_cmd(), "ffmpeg:dec", capture_out=True)
                time.sleep(0.5)
                if dec.poll() is not None:
                    raise RuntimeError(f"decoder quit instantly (code {dec.returncode})")

                # ── fire up audio sink ───────────────────────────────
                audio = self._spawn(self._audio_cmd(), "ffplay:audio")
                time.sleep(0.2)

                logging.info("◀ receiving %dx%d + audio from %s", w, h, self.a.url)
                preview_loop(dec, w, h, self.a.depth_loc, self.a.depthiness,
                             self.a.focus, self.a.diag)

            except KeyboardInterrupt:
                logging.info("interrupted — leaving receiver")
                break
            except Exception as exc:
                logging.error("recv error: %s", exc)
                logging.info("retrying in %d s", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
            finally:
                for p in (dec, audio):
                    if p and p.poll() is None:
                        p.terminate()
                        try:
                            p.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            p.kill()
                backoff = 1


# ════════════════════════════════════════════════════════════════════
#                           StreamSender
# ════════════════════════════════════════════════════════════════════
class StreamSender(_FFmpegMixin):
    def __init__(self, args):
        logging.info("Stream Sender created")
        self.a = args
        if args.camera:
            self.dshow_arg = self._resolve_camera_arg(args.camera)
            probe = self._probe_ffprobe(
                [
                    "ffprobe", "-v", "error", "-f", "dshow",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=s=x:p=0", "-i", self.dshow_arg,
                ],
                timeout=5.0,
            )
            self.width, self.height = probe if probe else (1280, 720)
            logging.info("camera resolution assumed %dx%d", self.width, self.height)
        else:
            if not args.video:
                raise SystemExit("You must supply a video path or --camera.")
            self.dshow_arg = None
            self.width, self.height = self._probe_ffprobe(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=p=0", args.video,
                ],
                timeout=None,
            ) or (1280, 720)

    # ───── camera index → name ───────────────────────────────────────
    @classmethod
    def _resolve_camera_arg(cls, cam: str) -> str:
        if cam.isdigit():
            idx, devices = int(cam), cls._list_dshow_video_devices()
            if not devices:
                raise SystemExit("No DirectShow video devices detected.")
            if idx >= len(devices):
                raise SystemExit(f"Camera index {idx} out of range (0-{len(devices)-1}).")
            logging.info('camera index %d → "%s"', idx, devices[idx])
            return f"video={devices[idx]}"
        return f"video={cam}"

    # ───── ffmpeg command builders ───────────────────────────────────
    def _dshow_input(self, retry_without_pixfmt: bool = False) -> list[str]:
        a = self.a
        size = a.cam_size
        fps  = str(a.cam_fps)
        video_size = ["-video_size", size] if size else []
        framerate  = ["-framerate", fps]  if fps else []
        pixf = (["-pixel_format", a.cam_pixfmt]
                if a.cam_pixfmt and not retry_without_pixfmt else [])
        return (["-f", "dshow", "-thread_queue_size", "1024"] +
                video_size + framerate + pixf + ["-i", self.dshow_arg])

    def _file_input(self) -> list[str]:
        return ["-re", "-stream_loop", "-1", "-i", self.a.video]

    def _enc_cmd(self, retry_without_pixfmt: bool = False) -> list[str]:
        a, url = self.a, self._udp_url(self.a.url)
        input_part = (self._dshow_input(retry_without_pixfmt)
                      if self.dshow_arg else self._file_input())

        base = ["-g", str(a.gop),
                "-force_key_frames", f"expr:gte(t,n_forced*{a.gop/30:.2f})"]

        if a.nvenc:
            base += ["-c:v", "h264_nvenc",
                     "-pix_fmt", "yuv444p" if a.yuv444 else "yuv420p",
                     "-preset", "p1", "-tune", "ull", "-zerolatency", "1"]
        else:
            base += ["-c:v", "libx264rgb", "-preset", "veryfast",
                     "-tune", "zerolatency", "-pix_fmt", "rgb24"]

        base += ["-b:v", f"{a.bitrate}k", "-maxrate", f"{a.bitrate}k",
                 "-minrate", f"{a.bitrate}k", "-bufsize", f"{a.bitrate//2}k"]

        muxer = "prompeg" if a.fec else "mpegts"
        if a.fec:
            url = url if url.startswith("prompeg+") else f"prompeg+{url}"
            base += ["-fec", "prompeg=8:4"]

        return ["ffmpeg", "-v", "error", *input_part, *base, "-f", muxer, url]

    # ───── decoder cmd for local preview of *files* ───────────────────
    def _dec_cmd_preview_file(self) -> list[str]:
        return ["ffmpeg", "-v", "warning", "-fflags", "nobuffer", "-flags", "low_delay",
                "-err_detect", "ignore_err+crccheck", "-i", self.a.video,
                "-vf", "format=rgba", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]

    # ───── public run() ───────────────────────────────────────────────
    def run(self) -> None:
        a = self.a
        src = self.dshow_arg or os.path.basename(a.video)
        logging.info("▶ streaming %s  %dx%d → %s", src, self.width, self.height, a.url)

        enc = self._spawn(self._enc_cmd(), "ffmpeg:enc")
        time.sleep(0.5)

        if enc.poll() is not None and self.dshow_arg and a.cam_pixfmt:
            logging.warning("capture failed with pixel-format \"%s\" — retrying without it",
                            a.cam_pixfmt)
            enc = self._spawn(self._enc_cmd(retry_without_pixfmt=True), "ffmpeg:enc")

        dec: Optional[subprocess.Popen] = None
        try:
            if a.quiet:
                enc.wait()
            else:
                if self.dshow_arg:
                    time.sleep(0.5)  # let packets arrive
                    dec = self._spawn(
                        ["ffmpeg", "-v", "warning", "-fflags", "nobuffer", "-flags", "low_delay",
                         "-i", self._udp_url(a.url),
                         "-vf", "format=rgba", "-f", "rawvideo",
                         "-pix_fmt", "rgba", "pipe:1"],
                        "ffmpeg:preview", capture_out=True)
                else:
                    dec = self._spawn(self._dec_cmd_preview_file(),
                                      "ffmpeg:dec", capture_out=True)

                preview_loop(dec, self.width, self.height, a.depth_loc,
                             a.depthiness, a.focus, a.diag)

        except KeyboardInterrupt:
            logging.info("Ctrl-C — stopping sender")
        finally:
            for p in (enc, dec):
                if p and p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        p.kill()


__all__ = ["StreamSender", "StreamReceiver"]
