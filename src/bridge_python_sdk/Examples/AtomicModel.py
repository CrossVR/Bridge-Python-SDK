#!/usr/bin/env python3
# AtomicModel.py – glowing nucleus and animated electrons
# generated in one shot by giving o3 Pro from OpenAI the SolarSystem.py as an example with the prompt: "the code above renders an image using the bridge-python-sdk write another interesting sample scene which follows this same pattern"
import math, time, sys, os

# ------------------------------------------------------- path so we can import the local engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from OpenGL import GL

from Rendering.Shader import Shader
from Rendering.Mesh   import Mesh
from Rendering.Render import Render


# ------------------------------------------------------- reusable geometry helpers
def create_sphere(radius: float = 1.0, segments: int = 32):
    verts = []
    for i in range(segments + 1):
        lat   = math.pi * (-0.5 + i / segments)
        sl, cl = math.sin(lat), math.cos(lat)
        for j in range(segments + 1):
            lon   = 2 * math.pi * j / segments
            slon, clon = math.sin(lon), math.cos(lon)

            x, y, z = radius * cl * clon, radius * sl, radius * cl * slon
            verts.append([x, y, z, x / radius, y / radius, z / radius,
                          j / segments, i / segments])
    tris = []
    for i in range(segments):
        for j in range(segments):
            v0 =  i    * (segments + 1) + j
            v1 =  v0 + 1
            v2 = (i+1) * (segments + 1) + j
            v3 =  v2 + 1
            tris.extend(verts[v0]); tris.extend(verts[v2]); tris.extend(verts[v1])
            tris.extend(verts[v1]); tris.extend(verts[v2]); tris.extend(verts[v3])
    return np.asarray(tris, dtype=np.float32)


def create_ring(radius: float = 1.0, thickness: float = 0.05,
                major_segments: int = 64, minor_segments: int = 8):
    """Thin torus used for orbital paths"""
    verts = []
    for i in range(major_segments + 1):
        u = 2 * math.pi * i / major_segments
        cu, su = math.cos(u), math.sin(u)
        for j in range(minor_segments + 1):
            v = 2 * math.pi * j / minor_segments
            cv, sv = math.cos(v), math.sin(v)
            x = (radius + thickness * cv) * cu
            y = (radius + thickness * cv) * su
            z = thickness * sv
            nx, ny, nz = cv * cu, cv * su, sv
            verts.append([x, y, z, nx, ny, nz, i/major_segments, j/minor_segments])
    tris = []
    for i in range(major_segments):
        for j in range(minor_segments):
            v0 = i * (minor_segments + 1) + j
            v1 = v0 + 1
            v2 = v0 + minor_segments + 1
            v3 = v2 + 1
            tris.extend(verts[v0]); tris.extend(verts[v2]); tris.extend(verts[v1])
            tris.extend(verts[v1]); tris.extend(verts[v2]); tris.extend(verts[v3])
    return np.asarray(tris, dtype=np.float32)


def create_stars(count: int = 400, radius: float = 25.0):
    quad = [(-1,-1),(1,-1),(1,1),(-1,-1),(1,1),(-1,1)]
    verts = []
    for _ in range(count):
        t, p = np.random.uniform(0, 2*math.pi), np.random.uniform(0, math.pi)
        sx, sy, sz = radius*math.sin(p)*math.cos(t), radius*math.cos(p), radius*math.sin(p)*math.sin(t)
        brightness = np.random.uniform(0.5, 1.0)
        size = 0.12
        for qx,qy in quad:
            verts.extend([sx+qx*size, sy+qy*size, sz,
                          0.0, brightness, 0.0,
                          (qx+1)*0.5, (qy+1)*0.5])
    return np.asarray(verts, dtype=np.float32)


# ------------------------------------------------------- shaders
# shared stride:  vec3 pos | vec3 normal | vec2 uv   (8 floats)
vertex_common = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
"""

matrices_block = """
uniform mat4 u_mvp;
uniform mat4 u_model;
"""

# ---------- nucleus (thick glowing sphere)
nucleus_vs = vertex_common + matrices_block + """
out vec3 vPos;
out vec3 vNormal;
void main(){
    vec4 world = u_model * vec4(aPos,1.0);
    gl_Position = u_mvp * vec4(aPos,1.0);
    vPos = world.xyz;
    vNormal = mat3(u_model)*aNormal;
}
"""

nucleus_fs = """
#version 330 core
in vec3 vPos;
in vec3 vNormal;
uniform float u_time;
out vec4 FragColor;
void main(){
    float glow = 1.3 + 0.3*sin(u_time*4.0 + length(vPos)*6.0);
    vec3 base  = vec3(0.9,0.2,0.1);
    float rim  = pow(1.0 - dot(normalize(vNormal), normalize(-vPos)), 2.0);
    vec3 color = base*glow + vec3(1.0,0.6,0.2)*rim*1.5;
    FragColor  = vec4(color,1.0);
}
"""

# ---------- electrons (smaller spheres, shifting hue)
electron_vs = nucleus_vs

electron_fs = """
#version 330 core
in vec3 vPos;
in vec3 vNormal;
uniform float u_time;
out vec4 FragColor;

vec3 hsv2rgb(vec3 c){ // tiny helper
    vec3 K = vec3(1.0, 2.0/3.0, 1.0/3.0);
    vec3 p = abs(fract(c.xxx + K.xyz)*6.0 - 3.0);
    return c.z * mix( vec3(1.0), clamp(p-1.0,0.0,1.0), c.y);
}

void main(){
    float speed = 4.0;
    float hue = fract(u_time/speed + length(vPos)*0.3);
    vec3  color = hsv2rgb(vec3(hue, 0.8, 1.0));
    float rim   = 1.0 - dot(normalize(vNormal), normalize(-vPos));
    color *= 0.4 + 0.6*pow(rim, 1.5);
    FragColor = vec4(color, 1.0);
}
"""

# ---------- orbital ring (dim emissive)
ring_vs = vertex_common + matrices_block + """
out vec2 vUV;
void main(){
    gl_Position = u_mvp * vec4(aPos,1.0);
    vUV = aUV;
}
"""

ring_fs = """
#version 330 core
in vec2 vUV;
out vec4 FragColor;
void main(){
    float edge = abs(sin(vUV.x*3.1415));
    vec3 color = vec3(0.2,0.4,0.9)*edge*0.6;
    FragColor = vec4(color, 1.0);
}
"""

# ---------- stars (same sprite technique as SolarSystem sample)
star_vs = vertex_common + """
uniform mat4 u_mvp;  // NO model for stars – they are in world space already
out float vBrightness; out vec2 vUV;
void main(){
    gl_Position = u_mvp * vec4(aPos,1.0);
    vBrightness = aNormal.y;
    vUV = aUV;
}
"""

star_fs = """
#version 330 core
in float vBrightness;
in vec2  vUV;
uniform float u_time;
out vec4 FragColor;
void main(){
    float d = distance(vUV, vec2(0.5));
    if(d>0.5) discard;
    float pulse = 0.5 + 0.5*sin(u_time*3.0 + vBrightness*12.0);
    float alpha = (1.0-smoothstep(0.3,0.5,d))*pulse;
    vec3  color = vec3(1.0,0.97,0.9)*vBrightness*pulse;
    FragColor = vec4(color*alpha, alpha);
}
"""


# ------------------------------------------------------- matrices (same helpers you already have)
def translation(x,y,z): return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]],dtype=np.float32)
def rotation_y(a): c,s=math.cos(a),math.sin(a); return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]],dtype=np.float32)
def rotation_z(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.float32)
def scale(x,y,z):     return np.array([[x,0,0,0],[0,y,0,0],[0,0,z,0],[0,0,0,1]],dtype=np.float32)


# ------------------------------------------------------- main demo driver
def main():
    renderer = Render(lkg_size=25)  # legacy-Khronos-GL window helper

    # geometry ---------------------------------------------------
    sphere = create_sphere(1.0, 24)
    ring1  = create_ring(radius=2.5, thickness=0.03)
    ring2  = create_ring(radius=4.0, thickness=0.03)
    ring3  = create_ring(radius=5.5, thickness=0.03)
    stars  = create_stars(900, 22.0)

    stride = 8*4
    attribs = [(0,3,GL.GL_FLOAT,False,stride,0),
               (1,3,GL.GL_FLOAT,False,stride,3*4),
               (2,2,GL.GL_FLOAT,False,stride,6*4)]

    sphere_mesh = Mesh(sphere, attribs)
    ring1_mesh  = Mesh(ring1,  attribs)
    ring2_mesh  = Mesh(ring2,  attribs)
    ring3_mesh  = Mesh(ring3,  attribs)
    star_mesh   = Mesh(stars,  attribs)

    # shaders ----------------------------------------------------
    nucleus_shader  = Shader(nucleus_vs,  nucleus_fs)
    electron_shader = Shader(electron_vs, electron_fs)
    ring_shader     = Shader(ring_vs,     ring_fs)
    star_shader     = Shader(star_vs,     star_fs)

    # object handles --------------------------------------------
    star_handle     = renderer.add_object(star_mesh,     star_shader)
    ring1_handle    = renderer.add_object(ring1_mesh,    ring_shader)
    ring2_handle    = renderer.add_object(ring2_mesh,    ring_shader)
    ring3_handle    = renderer.add_object(ring3_mesh,    ring_shader)
    nucleus_handle  = renderer.add_object(sphere_mesh,   nucleus_shader)

    # a handful of electrons
    electron_handles = [renderer.add_object(sphere_mesh, electron_shader) for _ in range(6)]

    # GL state
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
                           GL.GL_ONE, GL.GL_ONE)
    GL.glEnable(GL.GL_MULTISAMPLE)

    start = time.time()
    last  = start

    while not renderer.should_close():
        now = time.time()
        elapsed, dt = now - start, now - last
        last = now

        # time uniforms -----------------------------------------
        nucleus_shader.use();  nucleus_shader.set_uniform('u_time', elapsed)
        electron_shader.use(); electron_shader.set_uniform('u_time', elapsed)
        star_shader.use();     star_shader.set_uniform('u_time', elapsed)

        # transformations ---------------------------------------
        renderer.update_model(star_handle, rotation_y(elapsed*0.05))

        renderer.update_model(nucleus_handle, scale(1.2,1.2,1.2) @ rotation_y(elapsed*0.7))

        # orbital rings just rotate around Z to give some depth wobble
        renderer.update_model(ring1_handle, rotation_z(0.3) @ rotation_y(elapsed*0.4))
        renderer.update_model(ring2_handle, rotation_z(0.6) @ rotation_y(-elapsed*0.3))
        renderer.update_model(ring3_handle, rotation_z(1.0) @ rotation_y(elapsed*0.2))

        # electrons ------------------------------------------------
        # level 1: 2 electrons
        lvl1_radius = 2.5
        for i in range(2):
            ang = elapsed*2.0 + i*math.pi
            pos = (lvl1_radius*math.cos(ang), lvl1_radius*math.sin(ang), 0)
            renderer.update_model(electron_handles[i],
                                  translation(*pos) @ scale(0.25,0.25,0.25))

        # level 2: 4 electrons (split into two tilted planes)
        lvl2_radius = 4.0
        for i in range(4):
            ang = elapsed*1.5 + i*math.pi/2
            x = lvl2_radius*math.cos(ang)
            y = 0.4*math.sin(ang*2.0)
            z = lvl2_radius*math.sin(ang)
            renderer.update_model(electron_handles[2+i],
                                  translation(x,y,z) @ scale(0.25,0.25,0.25))

        # render frame -------------------------------------------
        renderer.render_frame(dt)


if __name__ == '__main__':
    main()
