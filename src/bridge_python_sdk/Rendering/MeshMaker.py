#!/usr/bin/env python3
# MeshMaker.py – one-stop NumPy vertex/triangle generators plus subdivision
import math
import random
import numpy as np

class MeshMaker:
    """Factory class that bundles every mesh-generation routine used by the demos
    and provides optional midpoint-subdivision for higher-resolution geometry."""

    # ------------------------------------------------------------------------- utility
    _STRIDE = 8  # position(3) + normal(3) + uv(2)  → 8 floats per vertex

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # ------------------------------------------------------------------------- subdivision
    @staticmethod
    def subdivide_mesh(verts: np.ndarray, level: int = 1) -> np.ndarray:
        """Midpoint-subdivide a triangle list (stride = 8) `level` times.

        Parameters
        ----------
        verts  : ndarray
            Flat (N×8) float32 array produced by MeshMaker generators.
        level  : int
            Number of recursive subdivision passes.

        Returns
        -------
        ndarray
            New flat float32 array with 4**level more triangles.
        """
        if level < 1:
            return verts.astype(np.float32, copy=False)

        tri = verts.reshape(-1, 3, MeshMaker._STRIDE).astype(np.float32)
        for _ in range(level):
            new_tris = []
            for v0, v1, v2 in tri:
                m01 = (v0 + v1) * 0.5
                m12 = (v1 + v2) * 0.5
                m20 = (v2 + v0) * 0.5

                # Re-normalize averaged normals
                for v in (m01, m12, m20):
                    v[3:6] = MeshMaker._normalize(v[3:6])

                new_tris.extend([
                    v0,  m01, m20,   # tri 1
                    m01, v1,  m12,   # tri 2
                    m20, m12, v2,    # tri 3
                    m01, m12, m20    # center tri
                ])
            tri = np.asarray(new_tris, np.float32).reshape(-1, 3, MeshMaker._STRIDE)
        return tri.reshape(-1, MeshMaker._STRIDE)

    # ------------------------------------------------------------------------- cube
    @staticmethod
    def create_cube(size: float = 1.0) -> np.ndarray:
        """Unit cube centered at origin (36 verts, stride = 8)."""
        s = size * 0.5
        faces = [
            # +X
            ([ s, -s, -s], [1, 0, 0], [0, 0]),
            ([ s, -s,  s], [1, 0, 0], [1, 0]),
            ([ s,  s,  s], [1, 0, 0], [1, 1]),
            ([ s, -s, -s], [1, 0, 0], [0, 0]),
            ([ s,  s,  s], [1, 0, 0], [1, 1]),
            ([ s,  s, -s], [1, 0, 0], [0, 1]),
            # -X
            ([-s, -s,  s], [-1, 0, 0], [0, 0]),
            ([-s, -s, -s], [-1, 0, 0], [1, 0]),
            ([-s,  s, -s], [-1, 0, 0], [1, 1]),
            ([-s, -s,  s], [-1, 0, 0], [0, 0]),
            ([-s,  s, -s], [-1, 0, 0], [1, 1]),
            ([-s,  s,  s], [-1, 0, 0], [0, 1]),
            # +Y
            ([-s,  s, -s], [0, 1, 0], [0, 0]),
            ([ s,  s, -s], [0, 1, 0], [1, 0]),
            ([ s,  s,  s], [0, 1, 0], [1, 1]),
            ([-s,  s, -s], [0, 1, 0], [0, 0]),
            ([ s,  s,  s], [0, 1, 0], [1, 1]),
            ([-s,  s,  s], [0, 1, 0], [0, 1]),
            # -Y
            ([-s, -s,  s], [0, -1, 0], [0, 0]),
            ([ s, -s,  s], [0, -1, 0], [1, 0]),
            ([ s, -s, -s], [0, -1, 0], [1, 1]),
            ([-s, -s,  s], [0, -1, 0], [0, 0]),
            ([ s, -s, -s], [0, -1, 0], [1, 1]),
            ([-s, -s, -s], [0, -1, 0], [0, 1]),
            # +Z
            ([ s, -s,  s], [0, 0, 1], [0, 0]),
            ([-s, -s,  s], [0, 0, 1], [1, 0]),
            ([-s,  s,  s], [0, 0, 1], [1, 1]),
            ([ s, -s,  s], [0, 0, 1], [0, 0]),
            ([-s,  s,  s], [0, 0, 1], [1, 1]),
            ([ s,  s,  s], [0, 0, 1], [0, 1]),
            # -Z
            ([-s, -s, -s], [0, 0, -1], [0, 0]),
            ([ s, -s, -s], [0, 0, -1], [1, 0]),
            ([ s,  s, -s], [0, 0, -1], [1, 1]),
            ([-s, -s, -s], [0, 0, -1], [0, 0]),
            ([ s,  s, -s], [0, 0, -1], [1, 1]),
            ([-s,  s, -s], [0, 0, -1], [0, 1]),
        ]
        verts = []
        for pos, nrm, uv in faces:
            verts.extend([*pos, *nrm, *uv])
        return np.asarray(verts, np.float32)

    # ------------------- remaining generators unchanged (cave, crystal, …) ---
    # (Methods from the previous answer are included here without edits.)
    # ------------------------------------------------------------------------- cave
    @staticmethod
    def create_cave(radius: float = 15.0, slices: int = 64, stacks: int = 32) -> np.ndarray:
        verts = []
        for i in range(stacks):
            lat0 = math.pi * (i / stacks - 0.5)
            lat1 = math.pi * ((i + 1) / stacks - 0.5)
            y0, y1 = radius * math.sin(lat0), radius * math.sin(lat1)
            r0, r1 = radius * math.cos(lat0), radius * math.cos(lat1)
            for j in range(slices):
                lon0 = 2 * math.pi * j / slices
                lon1 = 2 * math.pi * (j + 1) / slices
                def to_cart(r, lon): return r * math.cos(lon), r * math.sin(lon)
                x00, z00 = to_cart(r0, lon0)
                x01, z01 = to_cart(r0, lon1)
                x10, z10 = to_cart(r1, lon0)
                x11, z11 = to_cart(r1, lon1)
                quad = [
                    (x00, y0, z00, lon0, lat0),
                    (x01, y0, z01, lon1, lat0),
                    (x10, y1, z10, lon0, lat1),
                    (x01, y0, z01, lon1, lat0),
                    (x11, y1, z11, lon1, lat1),
                    (x10, y1, z10, lon0, lat1)
                ]
                for x, y, z, lo, la in quad:
                    u = lo / (2 * math.pi)
                    v = 0.5 - la / math.pi
                    nx, ny, nz = -x / radius, -y / radius, -z / radius
                    verts.extend([x, y, z, nx, ny, nz, u, v])
        return np.asarray(verts, np.float32)

    # ------------------------------------------------------------------------- crystal
    @staticmethod
    def create_crystal(height: float = 2.0, radius: float = 0.3, sides: int = 6) -> np.ndarray:
        vs = []
        for i in range(sides + 1):
            a = 2 * math.pi * i / sides
            x, z = radius * math.cos(a), radius * math.sin(a)
            nx, nz = math.cos(a), math.sin(a)
            vs.append([x, -height / 2, z, nx, 0, nz, i / sides, 1])
            vs.append([x,  height / 2, z, nx, 0, nz, i / sides, 0])
        vs.append([0, -height / 2 - radius, 0, 0, -1, 0, 0.5, 1])
        vs.append([0,  height / 2 + radius, 0, 0,  1, 0, 0.5, 0])
        tris = []
        for i in range(sides):
            a0, a1, a2, a3 = i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 3
            tris += vs[a0] + vs[a2] + vs[a1] + vs[a1] + vs[a2] + vs[a3]
        bot, top = len(vs) - 2, len(vs) - 1
        for i in range(sides):
            b0, b1 = i * 2, ((i + 1) % sides) * 2
            tris += vs[bot] + vs[b1] + vs[b0]
            t0, t1 = i * 2 + 1, ((i + 1) % sides) * 2 + 1
            tris += vs[top] + vs[t0] + vs[t1]
        return np.asarray(tris, np.float32)

    # ------------------------------------------------------------------------- water plane
    @staticmethod
    def create_water_plane(size: float = 20.0, segments: int = 50) -> np.ndarray:
        verts, step = [], size / segments
        for i in range(segments + 1):
            for j in range(segments + 1):
                x = -size / 2 + i * step
                z = -size / 2 + j * step
                verts += [x, 0, z, 0, 1, 0, i / segments, j / segments]
        tris = []
        for i in range(segments):
            for j in range(segments):
                v0 = i * (segments + 1) + j
                v1, v2, v3 = v0 + 1, v0 + segments + 1, v0 + segments + 2
                tris += verts[v0 * 8:v0 * 8 + 8] + verts[v2 * 8:v2 * 8 + 8] + verts[v1 * 8:v1 * 8 + 8]
                tris += verts[v1 * 8:v1 * 8 + 8] + verts[v2 * 8:v2 * 8 + 8] + verts[v3 * 8:v3 * 8 + 8]
        return np.asarray(tris, np.float32)

    # ------------------------------------------------------------------------- dust quads
    @staticmethod
    def create_dust(count: int = 500, bounds: float = 12.0) -> np.ndarray:
        quad = [(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)]
        vs = []
        for _ in range(count):
            x = np.random.uniform(-bounds, bounds)
            y = np.random.uniform(-2, 8)
            z = np.random.uniform(-bounds, bounds)
            sp = np.random.uniform(0.5, 2)
            ph = np.random.uniform(0, 2 * math.pi)
            br = np.random.uniform(0.3, 1)
            s = 0.05
            for qx, qy in quad:
                vs += [x + qx * s, y + qy * s, z, sp, ph, br, (qx + 1) * 0.5, (qy + 1) * 0.5]
        return np.asarray(vs, np.float32)

    # ------------------------------------------------------------------------- sphere
    @staticmethod
    def create_sphere(radius: float = 1.0, segments: int = 32) -> np.ndarray:
        vertices = []
        for i in range(segments + 1):
            lat = math.pi * (-0.5 + i / segments)
            sin_lat = math.sin(lat)
            cos_lat = math.cos(lat)
            for j in range(segments + 1):
                lon = 2 * math.pi * j / segments
                sin_lon = math.sin(lon)
                cos_lon = math.cos(lon)
                x = radius * cos_lat * cos_lon
                y = radius * sin_lat
                z = radius * cos_lat * sin_lon
                vertices.append([x, y, z, x / radius, y / radius, z / radius, j / segments, i / segments])
        tris = []
        for i in range(segments):
            for j in range(segments):
                v0 = i * (segments + 1) + j
                v1 = v0 + 1
                v2 = v0 + segments + 1
                v3 = v2 + 1
                tris += vertices[v0] + vertices[v2] + vertices[v1] + vertices[v1] + vertices[v2] + vertices[v3]
        return np.asarray(tris, np.float32)

    # ------------------------------------------------------------------------- torus
    @staticmethod
    def create_torus(major_radius: float = 1.0, minor_radius: float = 0.3, major_segments: int = 32, minor_segments: int = 16) -> np.ndarray:
        vertices = []
        for i in range(major_segments + 1):
            u = 2 * math.pi * i / major_segments
            cos_u = math.cos(u)
            sin_u = math.sin(u)
            for j in range(minor_segments + 1):
                v = 2 * math.pi * j / minor_segments
                cos_v = math.cos(v)
                sin_v = math.sin(v)
                x = (major_radius + minor_radius * cos_v) * cos_u
                y = minor_radius * sin_v
                z = (major_radius + minor_radius * cos_v) * sin_u
                nx = cos_v * cos_u
                ny = sin_v
                nz = cos_v * sin_u
                vertices.append([x, y, z, nx, ny, nz, i / major_segments, j / minor_segments])
        tris = []
        for i in range(major_segments):
            for j in range(minor_segments):
                v0 = i * (minor_segments + 1) + j
                v1 = v0 + 1
                v2 = v0 + minor_segments + 1
                v3 = v2 + 1
                tris += vertices[v0] + vertices[v2] + vertices[v1] + vertices[v1] + vertices[v2] + vertices[v3]
        return np.asarray(tris, np.float32)

    # ------------------------------------------------------------------------- star sprites
    @staticmethod
    def create_stars(count: int = 300, radius: float = 100.0) -> np.ndarray:
        vertices = []
        quad = [(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)]
        for _ in range(count):
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)
            brightness = random.uniform(0.6, 1.0)
            size = 0.12
            for qx, qy in quad:
                px = x + qx * size
                py = y + qy * size
                pz = z
                vertices.extend([px, py, pz, 0.0, brightness, 0.0, (qx + 1) * 0.5, (qy + 1) * 0.5])
        return np.asarray(vertices, np.float32)
