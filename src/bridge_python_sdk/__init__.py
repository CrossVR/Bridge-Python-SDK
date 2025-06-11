# src/bridge_python_sdk/__init__.py
"""
Bridge Python SDK – runtime package
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)           # resolve from installed wheel
except PackageNotFoundError:                  # editable/dev install fallback
    __version__ = "0.0.0+editable"

# # re-export public subpackages if you like:
# from .BridgeApi import *
# from .BridgeDataTypes import *
