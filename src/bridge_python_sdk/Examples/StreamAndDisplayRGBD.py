#!/usr/bin/env python3
# StreamAndDisplayRGBD.py — CLI wrapper; streaming logic lives in StreamSender / StreamReceiver

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Rendering.Stream import *

def send_command(args): StreamSender(args).run()
def recv_command(args): StreamReceiver(args).run()


def main():
    parser = argparse.ArgumentParser("SBS RGB-D streamer (OpenGL preview)")
    parser.add_argument("--log", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("send")
    s.add_argument("video")
    s.add_argument("--url", default="udp://127.0.0.1:5000")
    s.add_argument("--gop", type=int, default=30)
    s.add_argument("--bitrate", type=int, default=8000)
    s.add_argument("--fec", action="store_true")
    s.add_argument("--nvenc", action="store_true", default=True)
    s.add_argument("--yuv444", action="store_true")
    s.add_argument("-q", "--quiet", action="store_true")
    s.add_argument("--depthiness", type=float, default=1.0)
    s.add_argument("--focus", type=float, default=0.0)
    s.add_argument("--depth-loc", type=int, default=2)
    s.add_argument("--diag", action="store_true")

    r = sub.add_parser("recv")
    r.add_argument("--url", default="udp://localhost:5000")
    r.add_argument("--width", type=int)
    r.add_argument("--height", type=int)
    r.add_argument("--wait", type=float, default=30)
    r.add_argument("--depthiness", type=float, default=1.0)
    r.add_argument("--focus", type=float, default=0.0)
    r.add_argument("--depth-loc", type=int, default=2)
    r.add_argument("--diag", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log),
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")

    (send_command if args.cmd == "send" else recv_command)(args)


if __name__ == "__main__":
    main()
