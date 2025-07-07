
import math
import time
import sys
import os
from typing import Tuple

# --------------------------------------------------------------------------- #
#  Make our local rendering engine (Render, Mesh, Shader, etc.) importable
# --------------------------------------------------------------------------- #
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from OpenGL import GL

from Rendering.Shader import Shader
from Rendering.Mesh   import Mesh
from Rendering.Render import Render


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def create_sphere(radius: float = 1.0, segments: int = 32) -> np.ndarray:
    """Return a flat interleaved vertex array for a UV-sphere."""
    verts: list[list[float]] = []

    for i in range(segments + 1):
        lat = math.pi * (-0.5 + i / segments)
        sl, cl = math.sin(lat), math.cos(lat)

        for j in range(segments + 1):
            lon = 2.0 * math.pi * j / segments
            slon, clon = math.sin(lon), math.cos(lon)

            x, y, z = radius * cl * clon, radius * sl, radius * cl * slon
            verts.append([
                x, y, z,                        # position
                x / radius, y / radius, z / radius,  # normal
                j / segments, i / segments      # UV
            ])

    tri_stream: list[float] = []
    for i in range(segments):
        for j in range(segments):
            a =  i      * (segments + 1) + j
            b =  a + 1
            c = (i + 1) * (segments + 1) + j
            d =  c + 1

            tri_stream.extend(verts[a]); tri_stream.extend(verts[c]); tri_stream.extend(verts[b])
            tri_stream.extend(verts[b]); tri_stream.extend(verts[c]); tri_stream.extend(verts[d])

    return np.asarray(tri_stream, dtype=np.float32)


def create_ring(
    radius: float = 1.0,
    thickness: float = 0.05,
    major_segments: int = 64,
    minor_segments: int = 8
) -> np.ndarray:
    """Return a thin-torus vertex buffer suitable for an emissive ‘orbit’."""
    verts: list[list[float]] = []

    for i in range(major_segments + 1):
        u = 2.0 * math.pi * i / major_segments
        cu, su = math.cos(u), math.sin(u)

        for j in range(minor_segments + 1):
            v = 2.0 * math.pi * j / minor_segments
            cv, sv = math.cos(v), math.sin(v)

            x = (radius + thickness * cv) * cu
            y = (radius + thickness * cv) * su
            z = thickness * sv

            nx, ny, nz = cv * cu, cv * su, sv

            verts.append([
                x, y, z,
                nx, ny, nz,
                i / major_segments, j / minor_segments
            ])

    tri_stream: list[float] = []
    for i in range(major_segments):
        for j in range(minor_segments):
            a = i * (minor_segments + 1) + j
            b = a + 1
            c = a + (minor_segments + 1)
            d = c + 1

            tri_stream.extend(verts[a]); tri_stream.extend(verts[c]); tri_stream.extend(verts[b])
            tri_stream.extend(verts[b]); tri_stream.extend(verts[c]); tri_stream.extend(verts[d])

    return np.asarray(tri_stream, dtype=np.float32)


def create_stars(count: int = 400, radius: float = 25.0) -> np.ndarray:
    """Generate a simple sprite-quad star field."""
    quad: list[Tuple[int, int]] = [(-1, -1), (1, -1), (1, 1),
                                   (-1, -1), (1, 1), (-1, 1)]
    verts: list[float] = []

    for _ in range(count):
        theta, phi = np.random.uniform(0, 2.0 * math.pi), np.random.uniform(0, math.pi)
        sx = radius * math.sin(phi) * math.cos(theta)
        sy = radius * math.cos(phi)
        sz = radius * math.sin(phi) * math.sin(theta)

        brightness = np.random.uniform(0.5, 1.0)
        size = 0.12

        for qx, qy in quad:
            verts.extend([
                sx + qx * size,
                sy + qy * size,
                sz,
                0.0, brightness, 0.0,       # pack brightness in normal.y
                (qx + 1) * 0.5,
                (qy + 1) * 0.5
            ])

    return np.asarray(verts, dtype=np.float32)


# --------------------------------------------------------------------------- #
#  GLSL shader sources
# --------------------------------------------------------------------------- #
vertex_common = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
"""

# nucleus
nucleus_vs = vertex_common + """
uniform mat4 u_mvp;
out vec3 vPos;
out vec3 vNormal;
void main()
{
    gl_Position = u_mvp * vec4(aPos, 1.0);
    vPos        = aPos;
    vNormal     = aNormal;
}
"""

nucleus_fs = """
#version 330 core
in vec3 vPos;
in vec3 vNormal;
uniform float u_time;
out vec4 FragColor;
void main()
{
    float glow  = 1.3 + 0.3 * sin(u_time * 4.0 + length(vPos) * 6.0);
    vec3  base  = vec3(0.9, 0.2, 0.1);
    float rim   = pow(1.0 - dot(normalize(vNormal), normalize(-vPos)), 2.0);
    vec3  color = base * glow + vec3(1.0, 0.6, 0.2) * rim * 1.5;
    FragColor   = vec4(color, 1.0);
}
"""

# electrons
electron_vs = nucleus_vs

electron_fs = """
#version 330 core
in vec3 vPos;
in vec3 vNormal;
uniform float u_time;
out vec4 FragColor;

vec3 hsv2rgb(vec3 c)
{
    vec3 K = vec3(1.0, 2.0 / 3.0, 1.0 / 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

void main()
{
    float hue   = fract(u_time / 4.0 + length(vPos) * 0.3);
    vec3  color = hsv2rgb(vec3(hue, 0.8, 1.0));
    float rim   = 1.0 - dot(normalize(vNormal), normalize(-vPos));
    color      *= 0.4 + 0.6 * pow(rim, 1.5);
    FragColor   = vec4(color, 1.0);
}
"""

# ring
ring_vs = vertex_common + """
uniform mat4 u_mvp;
out vec2 vUV;
void main()
{
    gl_Position = u_mvp * vec4(aPos, 1.0);
    vUV         = aUV;
}
"""

ring_fs = """
#version 330 core
in vec2 vUV;
out vec4 FragColor;
void main()
{
    float edge  = abs(sin(vUV.x * 3.14159265));
    vec3  color = vec3(0.2, 0.4, 0.9) * edge * 0.6;
    FragColor   = vec4(color, 1.0);
}
"""

# stars
star_vs = vertex_common + """
uniform mat4 u_mvp;
out float vBrightness;
out vec2  vUV;
void main()
{
    gl_Position = u_mvp * vec4(aPos, 1.0);
    vBrightness = aNormal.y;
    vUV         = aUV;
}
"""

star_fs = """
#version 330 core
in float vBrightness;
in vec2  vUV;
uniform float u_time;
out vec4 FragColor;
void main()
{
    float d = distance(vUV, vec2(0.5));
    if (d > 0.5) discard;
    float pulse = 0.5 + 0.5 * sin(u_time * 3.0 + vBrightness * 12.0);
    float alpha = (1.0 - smoothstep(0.3, 0.5, d)) * pulse;
    vec3  color = vec3(1.0, 0.97, 0.9) * vBrightness * pulse;
    FragColor   = vec4(color * alpha, alpha);
}
"""


# --------------------------------------------------------------------------- #
#  Small matrix helpers
# --------------------------------------------------------------------------- #
def translation(x: float, y: float, z: float) -> np.ndarray:
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=np.float32)


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[ c, 0,  s, 0],
                     [ 0, 1,  0, 0],
                     [-s, 0,  c, 0],
                     [ 0, 0,  0, 1]], dtype=np.float32)


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[ c, -s, 0, 0],
                     [ s,  c, 0, 0],
                     [ 0,  0, 1, 0],
                     [ 0,  0, 0, 1]], dtype=np.float32)


def scale(x: float, y: float, z: float) -> np.ndarray:
    return np.array([[x, 0, 0, 0],
                     [0, y, 0, 0],
                     [0, 0, z, 0],
                     [0, 0, 0, 1]], dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Main scene driver
# --------------------------------------------------------------------------- #
def main() -> None:
    renderer = Render(lkg_size=25.0)

    # ---------------------- create geometry ------------------------------- #
    sphere      = create_sphere(segments=24)
    ring1_geom  = create_ring(radius=2.5, thickness=0.03)
    ring2_geom  = create_ring(radius=4.0, thickness=0.03)
    ring3_geom  = create_ring(radius=5.5, thickness=0.03)
    stars_geom  = create_stars(count=900, radius=22.0)

    stride  = 8 * 4  # eight floats * 4 bytes
    attribs = [
        (0, 3, GL.GL_FLOAT, False, stride, 0),
        (1, 3, GL.GL_FLOAT, False, stride, 3 * 4),
        (2, 2, GL.GL_FLOAT, False, stride, 6 * 4),
    ]

    sphere_mesh = Mesh(sphere,     attribs)
    ring1_mesh  = Mesh(ring1_geom, attribs)
    ring2_mesh  = Mesh(ring2_geom, attribs)
    ring3_mesh  = Mesh(ring3_geom, attribs)
    stars_mesh  = Mesh(stars_geom, attribs)

    # ---------------------- compile shaders ------------------------------ #
    nucleus_shader  = Shader(nucleus_vs,  nucleus_fs)
    electron_shader = Shader(electron_vs, electron_fs)
    ring_shader     = Shader(ring_vs,     ring_fs)
    star_shader     = Shader(star_vs,     star_fs)

    # ---------------------- stage objects -------------------------------- #
    stars_handle    = renderer.add_object(stars_mesh,  star_shader)
    ring1_handle    = renderer.add_object(ring1_mesh,  ring_shader)
    ring2_handle    = renderer.add_object(ring2_mesh,  ring_shader)
    ring3_handle    = renderer.add_object(ring3_mesh,  ring_shader)
    nucleus_handle  = renderer.add_object(sphere_mesh, nucleus_shader)

    # six electrons : two on ring1, four on ring2
    electron_handles = [renderer.add_object(sphere_mesh, electron_shader) for _ in range(6)]

    # ---------------------- GL state tweaks ------------------------------ #
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFuncSeparate(
        GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
        GL.GL_ONE, GL.GL_ONE
    )
    GL.glEnable(GL.GL_MULTISAMPLE)

    # ---------------------- main loop ------------------------------------ #
    start_time = time.time()
    last_time  = start_time

    while not renderer.should_close():
        now         = time.time()
        elapsed     = now - start_time
        dt          = now - last_time
        last_time   = now

        # ---------- per-frame uniforms
        for sh in (nucleus_shader, electron_shader, star_shader):
            sh.use()
            sh.set_uniform('u_time', elapsed)

        # ---------- background stars
        renderer.update_model(stars_handle, rotation_y(elapsed * 0.05))

        # ---------- nucleus
        nucleus_tf = rotation_y(elapsed * 0.7) @ scale(1.2, 1.2, 1.2)
        renderer.update_model(nucleus_handle, nucleus_tf)

        # ---------- rings (tilt + slow axial spin)
        ring1_tf = rotation_z(0.30) @ rotation_y(elapsed * 0.40)
        ring2_tf = rotation_z(0.60) @ rotation_y(-elapsed * 0.30)
        ring3_tf = rotation_z(1.00) @ rotation_y(elapsed * 0.20)

        renderer.update_model(ring1_handle, ring1_tf)
        renderer.update_model(ring2_handle, ring2_tf)
        renderer.update_model(ring3_handle, ring3_tf)

        # ---------- electrons on ring1 (XY plane, then ring tilt applied)
        ring1_radius = 2.5
        for i in range(2):
            ang = elapsed * 2.0 + i * math.pi          # opposite points
            local_tf = translation(
                ring1_radius * math.cos(ang),
                ring1_radius * math.sin(ang),
                0.0
            ) @ scale(0.25, 0.25, 0.25)
            renderer.update_model(electron_handles[i], ring1_tf @ local_tf)

        # ---------- electrons on ring2 (also XY plane to match torus)
        ring2_radius = 4.0
        for i in range(4):
            ang = elapsed * 1.5 + i * (math.pi / 2.0)  # quarter points
            local_tf = translation(
                ring2_radius * math.cos(ang),
                ring2_radius * math.sin(ang),
                0.0
            ) @ scale(0.25, 0.25, 0.25)
            renderer.update_model(electron_handles[2 + i], ring2_tf @ local_tf)

        # ---------- draw the frame
        renderer.render_frame(dt)


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    main()