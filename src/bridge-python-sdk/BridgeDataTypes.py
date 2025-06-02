# BridgeDataTypes.py

import ctypes
from enum import IntEnum

# === Type aliases ===
Window = ctypes.c_uint32

# === Enums ===
class UniqueHeadIndices(IntEnum):
    FirstLookingGlassDevice = 0xFFFFFFFF

class PixelFormats(IntEnum):
    NoFormat     = 0x0
    RGB          = 0x1907
    RGBA         = 0x1908
    BGRA         = 0x80E1
    Red          = 0x1903
    RGB_DXT1     = 0x83F0
    RGBA_DXT5    = 0x83F3
    YCoCg_DXT5   = 0x01
    A_RGTC1      = 0x8DBB
    SRGB         = 0x8C41
    SRGB_A       = 0x8C43
    R32F         = 0x822E
    RGBA32F      = 0x8814

class MTLPixelFormats(IntEnum):
    RGBA8Unorm     = 70
    BGRA8Unorm     = 80
    RGBA16Float    = 77
    RGBA32Float    = 124
    R32Float       = 85
    RG32Float      = 87
    RGB10A2Unorm   = 24
    RG16Float      = 74
    RG16Unorm      = 62
    Depth32Float   = 252

# === Structures ===
class CalibrationSubpixelCell(ctypes.Structure):
    _fields_ = [
        ("ROffsetX", ctypes.c_float),
        ("ROffsetY", ctypes.c_float),
        ("GOffsetX", ctypes.c_float),
        ("GOffsetY", ctypes.c_float),
        ("BOffsetX", ctypes.c_float),
        ("BOffsetY", ctypes.c_float),
    ]

class Dim(ctypes.Structure):
    _fields_ = [
        ("Width", ctypes.c_uint64),
        ("Height", ctypes.c_uint64),
    ]

class LKGCalibration(ctypes.Structure):
    _fields_ = [
        ("Center", ctypes.c_float),
        ("Pitch", ctypes.c_float),
        ("Slope", ctypes.c_float),
        ("Width", ctypes.c_int32),
        ("Height", ctypes.c_int32),
        ("Dpi", ctypes.c_float),
        ("FlipX", ctypes.c_float),
        ("InvView", ctypes.c_int32),
        ("Viewcone", ctypes.c_float),
        ("Fringe", ctypes.c_float),
        ("CellPatternMode", ctypes.c_int32),
        # this pointer will later point to a float array of length `number_of_cells * 6`
        ("Cells", ctypes.POINTER(ctypes.c_float)),
    ]

class DefaultQuiltSettings(ctypes.Structure):
    _fields_ = [
        ("Aspect", ctypes.c_float),
        ("QuiltWidth", ctypes.c_int32),
        ("QuiltHeight", ctypes.c_int32),
        ("QuiltColumns", ctypes.c_int32),
        ("QuiltRows", ctypes.c_int32),
    ]

class WindowPos(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_int64),
        ("Y", ctypes.c_int64),
    ]

class DisplayInfo(ctypes.Structure):
    _fields_ = [
        ("DisplayId", ctypes.c_uint32),
        ("Serial", ctypes.c_wchar_p),
        ("Name", ctypes.c_wchar_p),
        ("Dimensions", Dim),
        ("HwEnum", ctypes.c_int32),
        ("Calibration", LKGCalibration),
        ("Viewinv", ctypes.c_int32),
        ("Ri", ctypes.c_int32),
        ("Bi", ctypes.c_int32),
        ("Tilt", ctypes.c_float),
        ("Aspect", ctypes.c_float),
        ("Fringe", ctypes.c_float),
        ("Subp", ctypes.c_float),
        ("Viewcone", ctypes.c_float),
        ("Center", ctypes.c_float),
        ("Pitch", ctypes.c_float),
        ("DefaultQuiltSettings", DefaultQuiltSettings),
        ("WindowPosition", WindowPos),
    ]
