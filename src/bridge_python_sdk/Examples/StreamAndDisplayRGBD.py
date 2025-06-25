#!/usr/bin/env python3
# StreamAndDisplayRGBD.py — thin CLI wrapper; heavy lifting is done in
# Rendering/Stream.py (StreamSender / StreamReceiver).

import argparse
import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from Rendering.Stream import StreamSender, StreamReceiver        # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser("SBS RGB-D UDP streamer (OpenGL preview)")
    sub    = parser.add_subparsers(dest="cmd", required=True)

    # ───────────────────────── sender ──────────────────────────
    send = sub.add_parser("send", help="stream a video file or DirectShow camera")

    send.add_argument("video", nargs="?",
                      help="video file (ignored when --camera is set)")
    send.add_argument("--camera", metavar="ID|NAME",
                      help="capture from DirectShow camera (numeric index or quoted name)")

    # camera-specific tuning
    send.add_argument("--cam-size",  metavar="WxH", default=None,
                      help="force camera resolution, e.g. 1920x1080")
    send.add_argument("--cam-fps",   metavar="N",   type=int,  default=None,
                      help="force camera frame-rate, e.g. 60")
    send.add_argument("--cam-pixfmt", metavar="FMT", default=None,
                      help="force camera pixel-format (yuyv422, nv12, rgb24 …)")

    # generic encoder/network flags
    send.add_argument("--url",      default="udp://127.0.0.1:5000")
    send.add_argument("--gop",      type=int, default=30)
    send.add_argument("--bitrate",  type=int, default=8000)
    send.add_argument("--fec",      action="store_true")
    send.add_argument("--nvenc",    action="store_true", default=True)
    send.add_argument("--yuv444",   action="store_true")
    send.add_argument("-q", "--quiet", action="store_true")

    # RGB-D preview tuning
    send.add_argument("--depthiness", type=float, default=1.0)
    send.add_argument("--focus",      type=float, default=0.0)
    send.add_argument("--depth-loc",  type=int,   default=2)
    send.add_argument("--diag",       action="store_true")

    # ───────────────────────── receiver ──────────────────────────
    recv = sub.add_parser("recv", help="receive and preview an RGB-D stream")
    recv.add_argument("--url",    default="udp://127.0.0.1:5000")
    recv.add_argument("--width",  type=int)
    recv.add_argument("--height", type=int)
    recv.add_argument("--wait",   type=float, default=30)

    recv.add_argument("--depthiness", type=float, default=1.0)
    recv.add_argument("--focus",      type=float, default=0.0)
    recv.add_argument("--depth-loc",  type=int,   default=2)
    recv.add_argument("--diag",       action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.cmd == "send":
        StreamSender(args).run()
    else:
        StreamReceiver(args).run()


if __name__ == "__main__":
    main()
