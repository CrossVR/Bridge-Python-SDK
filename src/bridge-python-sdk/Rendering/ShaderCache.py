#!/usr/bin/env python3
import os
import re

def _load_shader(path, included=None):
    if included is None:
        included = set()
    full = os.path.abspath(path)
    if full in included:
        return ""
    included.add(full)
    directory = os.path.dirname(full)
    src = []
    pattern = re.compile(r'^\s*#import\s+"?([^"\s]+)"?')
    with open(full, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                inc = m.group(1)
                inc_path = os.path.join(directory, inc)
                src.append(_load_shader(inc_path, included))
            else:
                src.append(line)
    return "".join(src)

BASE_DIR = os.path.dirname(__file__)
SHADERS_DIR = os.path.join(BASE_DIR, "shaders")
VERTEX_SHADER_SRC = _load_shader(os.path.join(SHADERS_DIR, "vertex.glsl"))
FRAGMENT_SHADER_SRC = _load_shader(os.path.join(SHADERS_DIR, "fragment.glsl"))


# TAA blend shader sources
TAA_VERTEX_SRC = """
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 TexCoord;
void main()
{
    TexCoord = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""
TAA_FRAGMENT_SRC = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D u_currTex;
uniform sampler2D u_historyTex;
uniform float u_alpha;
void main()
{
    vec4 curr = texture(u_currTex, TexCoord);
    vec4 hist = texture(u_historyTex, TexCoord);
    FragColor = mix(curr, hist, u_alpha);
}
"""
# Simple display shader to blit the final history texture
DISPLAY_VERTEX_SRC = TAA_VERTEX_SRC
DISPLAY_FRAGMENT_SRC = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D u_tex;
void main()
{
    FragColor = texture(u_tex, TexCoord);
}
"""