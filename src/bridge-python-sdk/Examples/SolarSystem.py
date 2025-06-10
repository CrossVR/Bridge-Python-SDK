#!/usr/bin/env python3
# SolarSystem.py – solid Sun, brighter planets, fixed stride use
import math
import time
import sys
import os

# ----------------------------------------------------------------------------- path setup
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

# ----------------------------------------------------------------------------- imports
import numpy as np
from OpenGL import GL

from Rendering.Shader import Shader
from Rendering.Mesh   import Mesh
from Rendering.Render import Render

# ----------------------------------------------------------------------------- geometry
def create_sphere(radius: float = 1.0, segments: int = 32):
    vertices = []
    for i in range(segments + 1):
        lat      = math.pi * (-0.5 + i / segments)
        sin_lat  = math.sin(lat)
        cos_lat  = math.cos(lat)
        for j in range(segments + 1):
            lon      = 2.0 * math.pi * j / segments
            sin_lon  = math.sin(lon)
            cos_lon  = math.cos(lon)
            x = radius * cos_lat * cos_lon
            y = radius * sin_lat
            z = radius * cos_lat * sin_lon
            vertices.append(
                [x, y, z, x / radius, y / radius, z / radius, j / segments, i / segments]
            )
    triangles = []
    for i in range(segments):
        for j in range(segments):
            v0 = i * (segments + 1) + j
            v1 = v0 + 1
            v2 = v0 + segments + 1
            v3 = v2 + 1
            triangles.extend(vertices[v0])
            triangles.extend(vertices[v2])
            triangles.extend(vertices[v1])
            triangles.extend(vertices[v1])
            triangles.extend(vertices[v2])
            triangles.extend(vertices[v3])
    return np.asarray(triangles, dtype=np.float32)


def create_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    major_segments: int = 32,
    minor_segments: int = 16
):
    vertices = []
    for i in range(major_segments + 1):
        u        = 2.0 * math.pi * i / major_segments
        cos_u    = math.cos(u)
        sin_u    = math.sin(u)
        for j in range(minor_segments + 1):
            v        = 2.0 * math.pi * j / minor_segments
            cos_v    = math.cos(v)
            sin_v    = math.sin(v)
            x = (major_radius + minor_radius * cos_v) * cos_u
            y = minor_radius * sin_v
            z = (major_radius + minor_radius * cos_v) * sin_u
            nx = cos_v * cos_u
            ny = sin_v
            nz = cos_v * sin_u
            vertices.append(
                [x, y, z, nx, ny, nz, i / major_segments, j / minor_segments]
            )
    triangles = []
    for i in range(major_segments):
        for j in range(minor_segments):
            v0 = i * (minor_segments + 1) + j
            v1 = v0 + 1
            v2 = v0 + minor_segments + 1
            v3 = v2 + 1
            triangles.extend(vertices[v0])
            triangles.extend(vertices[v2])
            triangles.extend(vertices[v1])
            triangles.extend(vertices[v1])
            triangles.extend(vertices[v2])
            triangles.extend(vertices[v3])
    return np.asarray(triangles, dtype=np.float32)


def create_stars(count: int = 300, radius: float = 100.0):
    vertices = []
    quad = [(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)]
    for _ in range(count):
        theta = np.random.uniform(0.0, 2.0 * math.pi)
        phi   = np.random.uniform(0.0, math.pi)
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.cos(phi)
        z = radius * math.sin(phi) * math.sin(theta)
        brightness = np.random.uniform(0.6, 1.0)
        size = 0.12
        for qx, qy in quad:
            px = x + qx * size
            py = y + qy * size
            pz = z
            vertices.extend(
                [px, py, pz, 0.0, brightness, 0.0, (qx + 1) * 0.5, (qy + 1) * 0.5]
            )
    return np.asarray(vertices, dtype=np.float32)

# ----------------------------------------------------------------------------- shaders
planet_vertex_shader = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 vNormal;
out vec3 vPos;
out vec2 vUV;

void main(){
    vec4 worldPos = u_model * vec4(aPos, 1.0);
    gl_Position   = u_mvp * vec4(aPos, 1.0);
    vNormal       = normalize(mat3(u_model) * aNormal);
    vPos          = worldPos.xyz;
    vUV           = aUV;
}
"""

planet_fragment_shader = """
#version 330 core
in vec3 vNormal;
in vec3 vPos;
in vec2 vUV;

out vec4 FragColor;

void main(){
    vec3 lightPos   = vec3(0.0);
    vec3 lightColor = vec3(1.0, 0.95, 0.8) * 2.0;

    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightPos - vPos);
    vec3 V = normalize(-vPos);
    vec3 R = reflect(-L, N);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(V, R), 0.0), 32.0);

    float band = fract(vUV.x * 3.0);
    vec3 baseColor = (band < 0.333) ? vec3(0.2, 0.5, 0.8)
                    : (band < 0.666) ? vec3(0.8, 0.3, 0.2)
                    :                 vec3(0.3, 0.7, 0.3);

    baseColor += 0.25 * sin(vUV.x * 20.0) * sin(vUV.y * 15.0);

    vec3 ambient  = 0.35 * baseColor;
    vec3 diffuse  = 1.5  * diff  * lightColor * baseColor;
    vec3 specular = 0.9  * spec  * lightColor;

    FragColor = vec4(ambient + diffuse + specular, 1.0);
}
"""

sun_vertex_shader = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 vNormal;
out vec3 vPos;

void main(){
    vec4 worldPos = u_model * vec4(aPos, 1.0);
    gl_Position   = u_mvp * vec4(aPos, 1.0);
    vPos          = worldPos.xyz;
    vNormal       = normalize(mat3(u_model) * aNormal);
}
"""

# --- Sun is fully opaque (alpha = 1.0)
sun_fragment_shader = """
#version 330 core
in vec3 vNormal;
in vec3 vPos;

uniform float u_time;

out vec4 FragColor;

void main(){
    vec3  V     = normalize(-vPos);
    float rim   = 1.0 - abs(dot(vNormal, V));
    float glow  = 1.2 + 0.4 * sin(u_time * 5.0 + rim * 10.0);
    vec3  color = vec3(1.0, 0.78, 0.25) * glow;
    FragColor   = vec4(color, 1.0);
}
"""

star_vertex_shader = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

uniform mat4 u_mvp;

out float vBrightness;
out vec2  vUV;

void main(){
    gl_Position  = u_mvp * vec4(aPos, 1.0);
    vBrightness  = aNormal.y;
    vUV          = aUV;
}
"""

star_fragment_shader = """
#version 330 core
in float vBrightness;
in vec2  vUV;

uniform float u_time;

out vec4 FragColor;

void main(){
    float d = distance(vUV, vec2(0.5));
    if (d > 0.5) discard;

    float pulse = 0.5 + 0.5 * sin(u_time * 4.0 + vBrightness * 20.0);
    float alpha = (1.0 - smoothstep(0.3, 0.5, d)) * pulse;
    vec3  color = vec3(1.0, 0.95, 0.9) * vBrightness * pulse;

    FragColor = vec4(color * alpha, alpha);
}
"""

# ----------------------------------------------------------------------------- matrices
def translation(x: float, y: float, z: float):
    return np.array(
        [[1, 0, 0, x],
         [0, 1, 0, y],
         [0, 0, 1, z],
         [0, 0, 0, 1]],
        dtype=np.float32
    )


def rotation_y(a: float):
    c = math.cos(a)
    s = math.sin(a)
    return np.array(
        [[c, 0, s, 0],
         [0, 1, 0, 0],
         [-s, 0, c, 0],
         [0, 0, 0, 1]],
        dtype=np.float32
    )


def rotation_x(a: float):
    c = math.cos(a)
    s = math.sin(a)
    return np.array(
        [[1, 0, 0, 0],
         [0, c, -s, 0],
         [0, s,  c, 0],
         [0, 0, 0, 1]],
        dtype=np.float32
    )


def scale(sx: float, sy: float, sz: float):
    return np.array(
        [[sx, 0,  0,  0],
         [0,  sy, 0,  0],
         [0,  0,  sz, 0],
         [0,  0,  0,  1]],
        dtype=np.float32
    )

# ----------------------------------------------------------------------------- main
def main():
    renderer = Render(lkg_size=25)

    sphere_verts = create_sphere(1.0, 24)
    torus_verts  = create_torus(3.0, 0.2, 32, 8)
    star_verts   = create_stars(1000, 25.0)

    stride = 8 * 4

    attribs = [
        (0, 3, GL.GL_FLOAT, False, stride, 0),
        (1, 3, GL.GL_FLOAT, False, stride, 3 * 4),
        (2, 2, GL.GL_FLOAT, False, stride, 6 * 4)
    ]

    sphere_mesh = Mesh(sphere_verts, attribs)
    torus_mesh  = Mesh(torus_verts,  attribs)
    star_mesh   = Mesh(star_verts,   attribs)

    planet_shader = Shader(planet_vertex_shader, planet_fragment_shader)
    sun_shader    = Shader(sun_vertex_shader,    sun_fragment_shader)
    star_shader   = Shader(star_vertex_shader,   star_fragment_shader)

    # --- draw order: opaque objects first, Sun last ---
    planet1_handle = renderer.add_object(sphere_mesh, planet_shader)
    planet2_handle = renderer.add_object(sphere_mesh, planet_shader)
    moon1_handle   = renderer.add_object(sphere_mesh, planet_shader)
    moon2_handle   = renderer.add_object(sphere_mesh, planet_shader)
    rings_handle   = renderer.add_object(torus_mesh,  planet_shader)
    stars_handle   = renderer.add_object(star_mesh,   star_shader)
    sun_handle     = renderer.add_object(sphere_mesh, sun_shader)

    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFuncSeparate(
        GL.GL_SRC_ALPHA,
        GL.GL_ONE_MINUS_SRC_ALPHA,
        GL.GL_ONE,
        GL.GL_ONE
    )
    GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
    GL.glEnable(GL.GL_MULTISAMPLE)

    start_time = time.time()
    last_time  = start_time

    while not renderer.should_close():
        now = time.time()
        elapsed = now - start_time
        dt      = now - last_time
        last_time = now

        sun_shader.use()
        sun_shader.set_uniform("u_time", elapsed)

        star_shader.use()
        star_shader.set_uniform("u_time", elapsed)

        # ----------------------------------------------------------------- transforms
        sun_transform = scale(2.0, 2.0, 2.0) @ rotation_y(elapsed * 0.5)
        renderer.update_model(sun_handle, sun_transform)

        p1_ang = elapsed * 1.0
        renderer.update_model(
            planet1_handle,
            translation(
                5.0 * math.cos(p1_ang),
                0.0,
                5.0 * math.sin(p1_ang)
            )
            @ rotation_y(elapsed * 2.0)
            @ scale(0.8, 0.8, 0.8)
        )

        p2_ang = elapsed * 0.6
        renderer.update_model(
            planet2_handle,
            translation(
                8.0 * math.cos(p2_ang),
                1.5 * math.sin(p2_ang * 2.0),
                8.0 * math.sin(p2_ang)
            )
            @ rotation_x(0.3)
            @ rotation_y(elapsed * 1.5)
            @ scale(1.2, 1.2, 1.2)
        )

        m1_ang = elapsed * 3.0
        m1_pos = np.array(
            [
                5.0 * math.cos(p1_ang) + 1.5 * math.cos(m1_ang),
                0.5 * math.sin(m1_ang),
                5.0 * math.sin(p1_ang) + 1.5 * math.sin(m1_ang)
            ]
        )
        renderer.update_model(
            moon1_handle,
            translation(*m1_pos) @ scale(0.3, 0.3, 0.3)
        )

        m2_ang = elapsed * 2.5
        m2_pos = np.array(
            [
                8.0 * math.cos(p2_ang) + 2.0 * math.cos(m2_ang),
                1.5 * math.sin(p2_ang * 2.0) + 0.5 * math.sin(m2_ang),
                8.0 * math.sin(p2_ang) + 2.0 * math.sin(m2_ang)
            ]
        )
        renderer.update_model(
            moon2_handle,
            translation(*m2_pos) @ scale(0.4, 0.4, 0.4)
        )

        renderer.update_model(
            rings_handle,
            translation(
                8.0 * math.cos(p2_ang),
                1.5 * math.sin(p2_ang * 2.0),
                8.0 * math.sin(p2_ang)
            )
            @ rotation_x(1.2)
            @ rotation_y(elapsed * 0.5)
        )

        renderer.update_model(
            stars_handle,
            rotation_y(elapsed * 0.1)
        )

        renderer.render_frame(dt)

if __name__ == "__main__":
    main()
