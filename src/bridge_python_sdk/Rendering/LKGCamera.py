# LKGCamera.py
import math
from typing import Optional, Tuple

import numpy as np


class LKGCamera:
    """Looking-Glass camera that generates per-view matrices for quilt rendering."""

    def __init__(
        self,
        size: float = 3.0,
        center: Optional[np.ndarray] = None,
        up: Optional[np.ndarray] = None,
        fov: float = 14.0,
        viewcone: float = 40.0,
        aspect: float = 1.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> None:
        self.size = size
        self.center = center if center is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up = up if up is not None else np.array([0.0, -1.0, 0.0], dtype=np.float32)
        self.fov = fov
        self.viewcone = viewcone
        self.aspect = aspect
        self.near = near
        self.far = far

    # ------------------------------------------------ core helpers
    def _camera_distance(self) -> float:
        return self.size / math.tan(math.radians(self.fov))

    def _camera_offset(self) -> float:
        return self._camera_distance() * math.tan(math.radians(self.viewcone))

    # ------------------------------------------------ matrix builders
    def _view_matrix(self, offset: float, invert: bool) -> np.ndarray:
        f = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        u_base = self.up.copy()
        if invert:
            u_base[1] = -u_base[1]
        u = u_base / np.linalg.norm(u_base)
        s = np.cross(f, u)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)

        m = np.eye(4, dtype=np.float32)
        m[0, 0], m[1, 0], m[2, 0] = s
        m[0, 1], m[1, 1], m[2, 1] = u
        m[0, 2], m[1, 2], m[2, 2] = -f
        m[0, 3] = offset
        m[2, 3] = -self._camera_distance()

        if not invert:
            m = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32) @ m
        return m

    def _projection_matrix(self) -> np.ndarray:
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        n, fp = self.near, self.far
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / self.aspect
        m[1, 1] = f
        m[2, 2] = (fp + n) / (n - fp)
        m[2, 3] = (2 * fp * n) / (n - fp)
        m[3, 2] = -1.0
        return m

    # ------------------------------------------------ public API
    def compute_view_projection_matrices(
        self,
        normalized_view: float,
        invert: bool,
        depthiness: float,
        focus: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        offset = -(normalized_view - 0.5) * depthiness * self._camera_offset()
        view = self._view_matrix(offset, invert)
        proj = self._projection_matrix()

        if invert:
            focus = -focus

        # Write to row-major slot (0,2) so that, after the transpose in glUniformMatrix4fv,
        # the value lands in column-major M31.
        distance_from_center = normalized_view - 0.5
        frustum_shift = distance_from_center * focus
        proj[0, 2] += (offset * 2.0 / (self.size * self.aspect)) + frustum_shift
        return view, proj
