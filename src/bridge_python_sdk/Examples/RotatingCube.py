#!/usr/bin/env python3
# FrameworkRotatingCube.py – UV-aware data-driven effects

import math
import time
import sys
import os
import json
import threading
import queue
from typing import Dict, Tuple, List

import numpy as np
from OpenGL import GL
import glfw

# ───────────────────────────────────── EFFECT TABLE 
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HOW TO ADD YOUR OWN EFFECT                                              ║
# ║  DO NOT REMOVE OR CHANGE THIS BLOCK!                                     ║
# ║  1.  Each entry in EFFECTS_TABLE is a Python dict with three string keys ║
# ║      ─ "name"           : free-form label, used only for readability     ║
# ║      ─ "vertex_deform"  : GLSL snippet executed inside the vertex shader ║
# ║      ─ "fragment_tint"  : GLSL *expression* (returns vec3) used for tint ║
# ║                                                                          ║
# ║  2.  Inside every vertex snippet you can READ                            ║
# ║         aPosition  (vec3)   original model position                      ║
# ║         aNormal    (vec3)   model normal                                 ║
# ║         aUV        (vec2)   cube-face UV (0-1)                           ║
# ║         u_time     (float)  seconds since program start                  ║
# ║                                                                          ║
# ║      and you must WRITE                                                  ║
# ║         p         (vec3)   move this to deform the mesh                  ║
# ║         maxScale  (float)  absolute value of your largest displacement   ║
# ║         aux       (float)  any per-vertex number you want in the colour  ║
# ║                                                                          ║
# ║      ▸ Always assign maxScale *before* offsetting p.                     ║
# ║      ▸ The framework computes vBlendWeight from |p-aPosition|/maxScale.  ║
# ║      ▸ Store a useful scalar in aux; the fragment shader receives it as  ║
# ║        auxValue and can map it to colour however you like.               ║
# ║                                                                          ║
# ║  3.  fragment_tint must be a single GLSL expression returning vec3.      ║
# ║      Allowed helpers: hueToRGB(), mix(), math functions, constants.      ║
# ║      Do **not** terminate the expression with a semicolon.               ║
# ║                                                                          ║
# ║  4.  Keyboard mapping: list index 0-9 ↔ number keys 1-0.                 ║
# ║                                                                          ║
# ║  Quick template to copy:                                                 ║
# ║  {                                                                       ║
# ║      "name": "My wobble",                                                ║
# ║      "vertex_deform": """                                                ║
# ║          maxScale = 0.25;                                                ║
# ║          float wob = sin(u_time*5.0 + aUV.x*10.0);                       ║
# ║          p += normalize(aPosition) * maxScale * wob;                     ║
# ║          aux = wob;                                                      ║
# ║      """,                                                                ║
# ║      "fragment_tint": """                                                ║
# ║          mix(vec3(0.0,0.2,1.0), vec3(1.0,0.8,0.0), 0.5+0.5*auxValue)     ║
# ║      """                                                                 ║
# ║  },                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#

EFFECTS_TABLE: List[Dict[str, str]] = [
    {
        "name": "🌠 Meteor Pulse",
        "vertex_deform": """
            maxScale = 0.4;
            float pulse = exp(-length(aPosition.xy) * 3.0) * sin(length(aPosition) * 15.0 - u_time * 8.0);
            p += normalize(aPosition) * maxScale * pulse;
            aux = pulse;
        """,
        "fragment_tint": "mix(vec3(0.2,0.0,0.4), vec3(1.0,0.6,0.1), 0.5 + 0.5*sin(length(vWorldPos)*15.0 - u_time*8.0))"
    },
    {
        "name": "🌌 Cosmic Noise",
        "vertex_deform": """
            maxScale = 0.45;
            float noise = hashNoise(aPosition * 50.0, u_time * 3.0);
            float wave = sin(length(aPosition) * 6.0 - u_time * 2.0);
            p += aNormal * maxScale * noise * wave;
            aux = noise * wave;
        """,
        "fragment_tint": "hueToRGB(fract(0.7 + 0.3*hashNoise(vWorldPos*50.0,u_time*3.0)))"
    },
    {
        "name": "🔥 Solar Flare",
        "vertex_deform": """
            maxScale = 0.5;
            float flare = pow(max(0.0, sin(length(aPosition)*10.0 - u_time*5.0)), 4.0);
            p += aNormal * maxScale * flare;
            aux = flare;
        """,
        "fragment_tint": "mix(vec3(1.0,0.3,0.0), vec3(1.0,1.0,0.6), pow(max(0.0,sin(length(vWorldPos)*10.0 - u_time*5.0)),4.0))"
    },
    {
        "name": "🌊 Tsunami Waves",
        "vertex_deform": """
            maxScale = 0.42;
            float wave = sin(aPosition.y * 12.0 - u_time * 7.0);
            p += normalize(aPosition) * maxScale * wave;
            aux = wave;
        """,
        "fragment_tint": "mix(vec3(0.0,0.2,0.5), vec3(0.6,0.9,1.0), 0.5 + 0.5*sin(vWorldPos.y*12.0 - u_time*7.0))"
    },
    {
        "name": "🌀 Black Hole Spin",
        "vertex_deform": """
            maxScale = 0.4;
            float angle = atan(aPosition.z, aPosition.x) * 5.0 - u_time * 4.0;
            float distortion = sin(angle + length(aPosition.xy) * 12.0);
            p += normalize(aPosition) * maxScale * distortion;
            aux = distortion;
        """,
        "fragment_tint": "hueToRGB(fract(0.1 + 0.4*sin(atan(vWorldPos.z,vWorldPos.x)*5.0 - u_time*4.0)))"
    },
    {
        "name": "❄️ Frostbite Fractals",
        "vertex_deform": """
            maxScale = 0.38;
            float frost = pow(abs(sin(length(aPosition.xy)*25.0 - u_time*1.5)),3.0)*exp(-length(aPosition.xy)*3.0);
            p += aNormal * maxScale * frost;
            aux = frost;
        """,
        "fragment_tint": "mix(vec3(0.7,0.9,1.0), vec3(0.2,0.3,0.8), pow(abs(sin(length(vWorldPos.xy)*25.0 - u_time*1.5)),3.0))"
    },
    {
        "name": "🍃 Nature's Breath",
        "vertex_deform": """
            maxScale = 0.35;
            float breeze = smoothstep(-0.5,1.0,sin(u_time*2.0 + dot(aPosition,vec3(2.0,5.0,3.0))));
            p += aNormal * maxScale * breeze;
            aux = breeze;
        """,
        "fragment_tint": "mix(vec3(0.0,0.5,0.1), vec3(0.8,1.0,0.5), smoothstep(-0.5,1.0,sin(u_time*2.0+dot(vWorldPos,vec3(2,5,3)))))"
    },
    {
        "name": "⚡ Electric Storm",
        "vertex_deform": """
            maxScale = 0.45;
            float electric = hashNoise(aPosition*60.0,u_time*15.0)*smoothstep(0.0,1.0,sin(u_time*20.0-length(aPosition)*20.0));
            p += normalize(aPosition)*maxScale*electric;
            aux = electric;
        """,
        "fragment_tint": "mix(vec3(0.0,0.0,0.1), vec3(0.2,0.9,1.0), hashNoise(vWorldPos*60.0,u_time*15.0))"
    },
    {
        "name": "🌋 Lava Surge",
        "vertex_deform": """
            maxScale = 0.5;
            float surge = smoothstep(0.3,1.0,sin(u_time*3.0 - length(aPosition)*8.0));
            p += aNormal * maxScale * surge;
            aux = surge;
        """,
        "fragment_tint": "mix(vec3(0.3,0.0,0.0), vec3(1.0,0.4,0.0), smoothstep(0.3,1.0,sin(u_time*3.0-length(vWorldPos)*8.0)))"
    },
    {
        "name": "🌈 Chromatic Ripple",
        "vertex_deform": """
            maxScale = 0.4;
            float ripple = sin(length(aPosition.xy)*18.0 - u_time*9.0)*exp(-length(aPosition.xy)*2.0);
            p += normalize(aPosition)*maxScale*ripple;
            aux = ripple;
        """,
        "fragment_tint": "hueToRGB(fract(0.5+0.5*sin(length(vWorldPos.xy)*18.0-u_time*9.0)))"
    }
]

# ───────────────────────────────────── stdin reader thread ─────────────────────────────────────
# Blocks on readline, pushes into mode_queue
_mode_queue: "queue.Queue[str]" = queue.Queue()

def _stdin_reader():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        _mode_queue.put(line)

threading.Thread(target=_stdin_reader, daemon=True).start()

# ───────────────────────────────────── engine path ─────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Rendering.Shader    import Shader
from Rendering.Mesh      import Mesh
from Rendering.Render    import Render
from Rendering.MeshMaker import MeshMaker

# ─────────────────────────────────────────────────────────────────────────────
#  Geometry – create a subdivided cube with (pos, normal, uv) at each vertex
# ─────────────────────────────────────────────────────────────────────────────
SUBDIVISION_LEVEL: int = 4
cube_vertices: np.ndarray = MeshMaker.subdivide_mesh(
    MeshMaker.create_cube(1.0),
    SUBDIVISION_LEVEL
)

BYTES_PER_FLOAT: int = 4
STRIDE_BYTES: int = 8 * BYTES_PER_FLOAT

VERTEX_ATTRIBUTES: Tuple[Tuple[int,int,int,bool,int,int],...] = (
    (0,3,GL.GL_FLOAT,False,STRIDE_BYTES,0),
    (1,3,GL.GL_FLOAT,False,STRIDE_BYTES,3*BYTES_PER_FLOAT),
    (2,2,GL.GL_FLOAT,False,STRIDE_BYTES,6*BYTES_PER_FLOAT)
)

# ─────────────────────────────────────────────────────────────────────────────
#  Shader builder – converts EFFECTS_TABLE into GLSL sources
# ─────────────────────────────────────────────────────────────────────────────
def build_shaders_from_table(table: List[Dict[str,str]])->Tuple[str,str]:
    # Vertex shader
    v_lines = [
        "#version 330 core",
        "layout(location = 0) in vec3 aPosition;",
        "layout(location = 1) in vec3 aNormal;",
        "layout(location = 2) in vec2 aUV;",
        "",
        "uniform mat4  u_mvp;",
        "uniform float u_time;",
        "uniform int   u_effectIndex;",
        "uniform bool  u_effectEnabled;",
        "",
        "#define PI 3.14159265359",
        "",
        "out vec3  vNormal;",
        "out float vBlendWeight;",
        "out float vAuxParam;",
        "out vec3 vWorldPos;",
        "",
        "float hashNoise(vec3 p, float t) {",
        "    return sin(dot(p, vec3(17.0, 59.0, 15.0)) + t);",
        "}",
        "",
        "void main() {",
        "    vec3  p        = aPosition;",
        "    float maxScale = 0.0;",
        "    float aux      = 0.0;",
        "",
        "    if (u_effectEnabled) {"
    ]
    for idx,e in enumerate(table):
        v_lines.append(f"        if (u_effectIndex == {idx}) {{\n{e['vertex_deform']}\n        }}\n")
    v_lines += [
        "    }",
        "    float displacement = length(p - aPosition);",
        "    float dispRatio    = (maxScale == 0.0) ? 0.0 : displacement / maxScale;",
        "    vBlendWeight       = pow(smoothstep(0.0, 1.0, dispRatio), 1.4);",
        "    vNormal   = aNormal;",
        "    vAuxParam = aux;",
        "    vWorldPos = (u_mvp * vec4(p, 1.0)).xyz;",
        "    gl_Position = u_mvp * vec4(p, 1.0);",
        "}"
    ]
    vertex_src = "\n".join(v_lines)

    # Fragment shader
    f_lines = [
        "#version 330 core",
        "",
        "in vec3  vNormal;",
        "in float vBlendWeight;",
        "in float vAuxParam;",
        "in vec3 vWorldPos;",
        "",
        "uniform mat4  u_mvp;",
        "uniform float u_time;",
        "uniform int   u_effectIndex;",
        "uniform bool  u_effectEnabled;",
        "",
        "out vec4 FragColor;",
        "",
        "#define PI 3.14159265359",
        "",
        "vec3 hueToRGB(float h) {",
        "    float R = abs(h * 6.0 - 3.0) - 1.0;",
        "    float G = 2.0 - abs(h * 6.0 - 2.0);",
        "    float B = 2.0 - abs(h * 6.0 - 4.0);",
        "    return clamp(vec3(R, G, B), 0.0, 1.0);",
        "}",
        "",
        "float hashNoise(vec3 p, float t) {",
        "    return sin(dot(p, vec3(17.0, 59.0, 15.0)) + t);",
        "}",
        "",
        "vec3 effectTint(int idx, float auxValue) {",
        "    vec3 tint = vec3(1.0);"
    ]
    for idx,e in enumerate(table):
        f_lines.append(f"    if (idx == {idx}) tint = {e['fragment_tint']};")
    f_lines += [
        "    return tint;",
        "}",
        "",
        "void main() {",
        "    vec3 baseColour = abs(vNormal);",
        "    if (u_effectEnabled) {",
        "        vec3 tint = effectTint(u_effectIndex, vAuxParam);",
        "        baseColour = mix(baseColour, tint, vBlendWeight);",
        "    }",
        "    FragColor = vec4(baseColour, 1.0);",
        "}"
    ]
    fragment_src = "\n".join(f_lines)
    return vertex_src, fragment_src

def q_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_n = axis / np.linalg.norm(axis)
    ha = angle_rad * 0.5
    s = math.sin(ha)
    return np.array([math.cos(ha), *(axis_n * s)], dtype=np.float32)

def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)

def q_to_mat4(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w), 0.0],
        [2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z - x*w), 0.0],
        [2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y),   0.0],
        [0.0,           0.0,           0.0,             1.0]
    ], dtype=np.float32)

def main() -> None:
    renderer = Render(debug=False)

    vert_src, frag_src = build_shaders_from_table(EFFECTS_TABLE)
    shader = Shader(vert_src, frag_src)

    cube_mesh   = Mesh(cube_vertices, VERTEX_ATTRIBUTES)
    cube_handle = renderer.add_object(cube_mesh, shader)

    quat            = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    idle_spin_axis  = np.array([0.3, 0.5, 0.4], dtype=np.float32)
    key_rot_speed   = 1.0
    mouse_sensitivity = 0.005
    idle_delay      = 3.0

    effect_idx      = 0
    effects_on      = True
    wireframe_on    = False
    spin_paused     = False
    anim_paused     = False

    keys_to_track = [
        glfw.KEY_F, glfw.KEY_P, glfw.KEY_R, glfw.KEY_SPACE,
        glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3,
        glfw.KEY_4, glfw.KEY_5, glfw.KEY_6, glfw.KEY_7,
        glfw.KEY_8, glfw.KEY_9
    ]
    NUMBER_KEY_MAP = {
        glfw.KEY_1:0, glfw.KEY_2:1, glfw.KEY_3:2,
        glfw.KEY_4:3, glfw.KEY_5:4, glfw.KEY_6:5,
        glfw.KEY_7:6, glfw.KEY_8:7, glfw.KEY_9:8,
        glfw.KEY_0:9
    }
    key_was_down = {k: False for k in keys_to_track}

    last_input_time = time.time()
    last_frame_time = last_input_time
    shader_clock    = 0.0
    mouse_prev      = None

    while not renderer.should_close():
        now = time.time()
        dt  = now - last_frame_time
        last_frame_time = now

        if not anim_paused:
            shader_clock += dt * 0.45

        # continuous rotation via WASDQE
        for key, (axis, sign) in {
            glfw.KEY_W: (np.array([1, 0, 0], dtype=np.float32), 1),
            glfw.KEY_S: (np.array([1, 0, 0], dtype=np.float32), -1),
            glfw.KEY_A: (np.array([0, 1, 0], dtype=np.float32), 1),
            glfw.KEY_D: (np.array([0, 1, 0], dtype=np.float32), -1),
            glfw.KEY_Q: (np.array([0, 0, 1], dtype=np.float32), 1),
            glfw.KEY_E: (np.array([0, 0, 1], dtype=np.float32), -1),
        }.items():
            if renderer.window.is_key_pressed(key):
                angle = key_rot_speed * sign * dt
                quat  = q_mul(quat, q_from_axis_angle(axis, angle))
                last_input_time = now

        # edge-detected keys
        for key in keys_to_track:
            is_down = renderer.window.is_key_pressed(key)
            if is_down and not key_was_down[key]:
                if key == glfw.KEY_F:
                    wireframe_on = not wireframe_on
                    GL.glPolygonMode(
                        GL.GL_FRONT_AND_BACK,
                        GL.GL_LINE if wireframe_on else GL.GL_FILL
                    )
                elif key == glfw.KEY_P:
                    spin_paused = not spin_paused
                    anim_paused = not anim_paused
                elif key == glfw.KEY_R:
                    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                elif key == glfw.KEY_SPACE:
                    effects_on = not effects_on
                elif glfw.KEY_1 <= key <= glfw.KEY_9 or key == glfw.KEY_0:
                    effect_idx = NUMBER_KEY_MAP[key]
                    sys.stdout.write(f"mode: {effect_idx}\n")
                    sys.stdout.flush()
                last_input_time = now
            key_was_down[key] = is_down

        # mouse-drag rotation
        if renderer.window.is_mouse_button_pressed(glfw.MOUSE_BUTTON_LEFT):
            mx, my = renderer.window.get_mouse_pos()
            if mouse_prev is not None:
                dx, dy = mx - mouse_prev[0], my - mouse_prev[1]
                quat = q_mul(
                    q_from_axis_angle(np.array([0,1,0],dtype=np.float32),
                                      dx * mouse_sensitivity),
                    quat
                )
                quat = q_mul(
                    q_from_axis_angle(np.array([1,0,0],dtype=np.float32),
                                      dy * mouse_sensitivity),
                    quat
                )
                last_input_time = now
            mouse_prev = (mx, my)
        else:
            mouse_prev = None

        # idle auto-spin
        if not spin_paused and (now - last_input_time > idle_delay):
            axis_norm = idle_spin_axis / np.linalg.norm(idle_spin_axis)
            angle     = np.linalg.norm(idle_spin_axis) * dt
            quat      = q_mul(q_from_axis_angle(axis_norm, angle), quat)

        # hot-reload JSON from stdin queue
        while not _mode_queue.empty():
            raw = _mode_queue.get()
            try:
                new_mode = json.loads(raw)
                idx = new_mode.get("index")
                if not isinstance(idx, int) or not (0 <= idx < len(EFFECTS_TABLE)):
                    raise ValueError("Invalid index")
                for k in ("name","vertex_deform","fragment_tint"):
                    if k not in new_mode:
                        raise ValueError(f"Missing {k}")
                EFFECTS_TABLE[idx] = {
                    "name": new_mode["name"],
                    "vertex_deform": new_mode["vertex_deform"],
                    "fragment_tint": new_mode["fragment_tint"]
                }
                vert_src, frag_src = build_shaders_from_table(EFFECTS_TABLE)
                shader = Shader(vert_src, frag_src)
                effect_idx = idx
                effects_on = True
                sys.stdout.write("SUCCESS\n")
                sys.stdout.flush()
            except Exception:
                sys.stdout.write("ERROR\n")
                sys.stdout.flush()

        # render
        shader.use()
        shader.set_uniform("u_time", shader_clock)
        shader.set_uniform("u_effectIndex", effect_idx)
        shader.set_uniform("u_effectEnabled", effects_on)
        renderer.update_model(cube_handle, q_to_mat4(quat))
        renderer.render_frame(dt)


if __name__ == "__main__":
    main()
