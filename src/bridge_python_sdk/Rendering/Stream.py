#!/usr/bin/env python3
# Rendering/Stream.py — streaming back-end for StreamAndDisplayRGBD CLI
# (full file – no omissions, now with early progress messages)

import logging
import os
import re
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import glfw
import numpy as np
from OpenGL import GL

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from BridgeApi import BridgeAPI, PixelFormats  # noqa: E402

# ════════════════════════════════════════════════════════════════════════
#                          ffmpeg helpers
# ════════════════════════════════════════════════════════════════════════
class _FFmpegMixin:
    _MIN_DIM     = 32
    _FIFO_BYTES  = 5_000_000
    _VIDEO_RE    = re.compile(r"Video:.*? (\d+)x(\d+)")
    _WXH_RE      = re.compile(r"(\d+)[x,](\d+)")

    # ---------- process spawning / logging ----------
    @staticmethod
    def _spawn(cmd, tag, capture_out=False):
        """Start *cmd* and route stderr lines to the Python logger."""
        logging.info("%s: exec %s", tag, " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if capture_out else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        threading.Thread(
            target=_FFmpegMixin._forward_log,
            args=(proc.stderr, tag),
            daemon=True,
        ).start()
        return proc

    @staticmethod
    def _forward_log(stream, prefix):
        for raw in iter(stream.readline, b""):
            if not raw:
                break
            logging.warning("%s | %s", prefix, raw.decode(errors="replace").rstrip())

    # ---------- URL helpers ----------
    @classmethod
    def _udp_url(cls, url):
        p = urlparse(url)
        if p.scheme not in ("udp", "prompeg+udp"):
            return url
        q = dict(parse_qsl(p.query))
        q.setdefault("fifo_size", str(cls._FIFO_BYTES))
        q.setdefault("overrun_nonfatal", "1")
        return urlunparse(p._replace(query=urlencode(q)))

    # ---------- probing helpers ----------
    @classmethod
    def _probe_file(cls, path):
        return cls._do_probe(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", path]
        )

    @classmethod
    def _probe_stream(cls, url, timeout_s):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", url]
        return cls._do_probe(cmd, timeout_s)

    @classmethod
    def _do_probe(cls, cmd, timeout_s=None):
        try:
            out = subprocess.check_output(cmd, text=True, timeout=timeout_s).strip()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return None
        for line in out.splitlines():
            if (m := cls._WXH_RE.search(line)):
                w, h = int(m[1]), int(m[2])
                if not w & 1 and w >= cls._MIN_DIM and h >= cls._MIN_DIM:
                    return w, h
        return None

    @classmethod
    def _probe_udp(cls, url, timeout_s):
        deadline = time.time() + timeout_s
        cmd = ["ffmpeg", "-v", "warning", "-fflags", "nobuffer", "-flags", "low_delay",
               "-i", cls._udp_url(url), "-frames:v", "1", "-f", "null", "-"]
        while time.time() < deadline:
            proc = cls._spawn(cmd, "probe:udp")
            for raw in proc.stderr:
                if (m := cls._VIDEO_RE.search(raw.decode(errors="ignore"))):
                    proc.kill()
                    w, h = int(m[1]), int(m[2])
                    if not w & 1 and w >= cls._MIN_DIM and h >= cls._MIN_DIM:
                        return w, h
                    return None
            time.sleep(0.4)
            proc.kill()
        return None


# ════════════════════════════════════════════════════════════════════════
#                    shared OpenGL preview loop
# ════════════════════════════════════════════════════════════════════════
def _norm_focus(focus, depth):
    return 0.005 + (((focus * depth) + 1) / 2) * (-0.007 - 0.005)


def preview_loop(proc, width, height, depth_loc, depth_scale, focus, diag):
    if not glfw.init():
        sys.exit("GLFW init failed")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(1, 1, "", None, None)
    glfw.make_context_current(win)

    bridge = BridgeAPI()
    bridge.initialize("RGBD")
    handle = bridge.instance_window_gl(-1)
    aspect, qw, qh, tiles_x, tiles_y = bridge.get_default_quilt_settings(handle)

    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8,
                    width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)

    bpf = width * height * 4
    buf, frames, t0 = bytearray(), 0, time.time()
    f_norm = _norm_focus(focus, depth_scale)

    while not glfw.window_should_close(win):
        while len(buf) < bpf and proc.poll() is None:
            chunk = proc.stdout.read(bpf - len(buf))
            if not chunk:
                time.sleep(0.002)
                continue
            buf.extend(chunk)
        if len(buf) < bpf:
            break

        raw = bytes(buf[:bpf])
        del buf[:bpf]

        try:
            rgba = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
        except ValueError:
            buf.clear()
            continue

        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0,
                           width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba)

        bridge.draw_interop_rgbd_texture_gl(
            handle, tex, PixelFormats.RGBA,
            width, height, qw, qh, tiles_x, tiles_y,
            float(aspect), f_norm, depth_scale, 1.0, depth_loc,
        )

        glfw.swap_buffers(win)
        glfw.poll_events()

        frames += 1
        if time.time() - t0 >= 1:
            logging.info("FPS %d", frames)
            if diag:
                logging.debug("sample RGBA %s", tuple(rgba[0, 0]))
            frames, t0 = 0, time.time()

    GL.glDeleteTextures(1, [tex])
    glfw.destroy_window(win)
    glfw.terminate()


# ════════════════════════════════════════════════════════════════════════
#                           StreamSender
# ════════════════════════════════════════════════════════════════════════
class StreamSender(_FFmpegMixin):
    def __init__(self, args):
        self.a = args
        self.width, self.height = self._probe_file(args.video)

    # ---------- encoder / preview commands ----------
    def _enc_cmd(self):
        a, url = self.a, self._udp_url(self.a.url)
        opts = [
            "-g", str(a.gop),
            "-force_key_frames", f"expr:gte(t,n_forced*{a.gop/30:.2f})"
        ]
        if a.nvenc:
            opts += [
                "-c:v", "h264_nvenc",
                "-pix_fmt", "yuv444p" if a.yuv444 else "yuv420p",
                "-preset", "p1", "-tune", "ull", "-zerolatency", "1",
            ]
        else:
            opts += [
                "-c:v", "libx264rgb", "-preset", "veryfast",
                "-tune", "zerolatency", "-pix_fmt", "rgb24",
            ]
        opts += [
            "-b:v", f"{a.bitrate}k",
            "-maxrate", f"{a.bitrate}k",
            "-minrate", f"{a.bitrate}k",
            "-bufsize", f"{a.bitrate // 2}k",
        ]
        muxer = "prompeg" if a.fec else "mpegts"
        if a.fec:
            url = url if url.startswith("prompeg+") else f"prompeg+{url}"
            opts += ["-fec", "prompeg=8:4"]

        return ["ffmpeg", "-v", "error", "-re", "-stream_loop", "-1", "-i", a.video,
                *opts, "-f", muxer, url]

    def _dec_cmd_preview(self):
        return [
            "ffmpeg", "-v", "warning", "-fflags", "nobuffer", "-flags", "low_delay",
            "-err_detect", "ignore_err+crccheck", "-i", self.a.video,
            "-vf", "format=rgba", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1",
        ]

    # ---------- run ----------
    def run(self):
        a = self.a
        logging.info("▶ streaming %s  %dx%d → %s",
                     os.path.basename(a.video), self.width, self.height, a.url)
        enc = self._spawn(self._enc_cmd(), "ffmpeg:enc")
        dec = None
        try:
            if a.quiet:
                enc.wait()
            else:
                dec = self._spawn(self._dec_cmd_preview(), "ffmpeg:dec", capture_out=True)
                preview_loop(dec, self.width, self.height,
                             a.depth_loc, a.depthiness, a.focus, a.diag)
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


# ════════════════════════════════════════════════════════════════════════
#                           StreamReceiver
# ════════════════════════════════════════════════════════════════════════
class StreamReceiver(_FFmpegMixin):
    def __init__(self, args):
        self.a = args

    # ---------- decoder command ----------
    def _dec_cmd(self):
        return [
            "ffmpeg", "-v", "warning", "-fflags", "nobuffer", "-flags", "low_delay",
            "-err_detect", "ignore_err+crccheck", "-i", self._udp_url(self.a.url),
            "-vf", "format=rgba", "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1",
        ]

    # ---------- run ----------
    def run(self):
        backoff = 1
        while True:
            dec = None
            try:
                # --- determine resolution --------------------------------
                if self.a.width and self.a.height:
                    logging.info("using user-supplied resolution %dx%d", self.a.width, self.a.height)
                    w, h = self.a.width, self.a.height
                else:
                    logging.info("probing with ffprobe …")
                    probe = self._probe_stream(self.a.url, self.a.wait)
                    if probe:
                        w, h = probe
                        logging.info("stream resolution %dx%d (via ffprobe)", w, h)
                    else:
                        logging.info("probing via UDP sniff …")
                        probe = self._probe_udp(self.a.url, self.a.wait)
                        if probe:
                            w, h = probe
                            logging.info("stream resolution %dx%d (via UDP sniff)", w, h)
                        else:
                            raise RuntimeError("cannot detect stream dimensions")

                # --- start decoder --------------------------------------
                logging.info("decoder URL: %s", self._udp_url(self.a.url))
                cmd = self._dec_cmd()
                dec = self._spawn(cmd, "ffmpeg:dec", capture_out=True)

                # quick check: did ffmpeg exit immediately?
                time.sleep(0.5)
                if dec.poll() is not None:
                    raise RuntimeError(f"ffmpeg exited instantly (code {dec.returncode})")

                logging.info("◀ receiving %dx%d from %s", w, h, self.a.url)
                preview_loop(
                    dec, w, h,
                    self.a.depth_loc, self.a.depthiness,
                    self.a.focus, self.a.diag,
                )
            except KeyboardInterrupt:
                logging.info("interrupted — exiting receiver")
                break
            except Exception as exc:
                logging.error("recv error: %s", exc)
                logging.info("retrying in %d s", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
            finally:
                if dec and dec.poll() is None:
                    dec.terminate()
                    try:
                        dec.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        dec.kill()
                logging.debug("decoder clean-up complete")
                logging.info("stream ended — reconnecting immediately")
                backoff = 1
