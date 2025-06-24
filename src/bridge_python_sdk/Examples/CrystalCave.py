#!/usr/bin/env python3
# CrystalCave.py – HDR-textured cave with clustered PBR crystals, animated water,
#                  floating dust, fully compatible with unmodified Render.py.
#
# ALL names remain exactly as declared; no abbreviations, no omissions.

import math, random, time, sys, os
import numpy as np
from OpenGL import GL
import glfw

# ----------------------------------------------------------------------------- local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Rendering.Shader   import Shader
from Rendering.Mesh     import Mesh
from Rendering.Render   import Render
from Rendering.Texture  import Texture2D

# ----------------------------------------------------------------------------- geometry helpers
def create_cave(radius: float = 15.0, slices: int = 64, stacks: int = 32):
    """Inward-facing sphere with latitude/longitude UVs (CW winding)."""
    verts = []
    for i in range(stacks):
        lat0 = math.pi * (i       / stacks - 0.5)
        lat1 = math.pi * ((i + 1) / stacks - 0.5)
        y0, y1 = radius*math.sin(lat0), radius*math.sin(lat1)
        r0, r1 = radius*math.cos(lat0), radius*math.cos(lat1)
        for j in range(slices):
            lon0 = 2*math.pi *  j      / slices
            lon1 = 2*math.pi * (j + 1) / slices
            def to_cart(r, lon): return r*math.cos(lon), r*math.sin(lon)
            x00,z00 = to_cart(r0, lon0)
            x01,z01 = to_cart(r0, lon1)
            x10,z10 = to_cart(r1, lon0)
            x11,z11 = to_cart(r1, lon1)
            quad = [ (x00,y0,z00, lon0,lat0),
                     (x01,y0,z01, lon1,lat0),
                     (x10,y1,z10, lon0,lat1),
                     (x01,y0,z01, lon1,lat0),
                     (x11,y1,z11, lon1,lat1),
                     (x10,y1,z10, lon0,lat1) ]
            for x,y,z,lo,la in quad:
                u = lo/(2*math.pi); v = 0.5 - la/math.pi
                nx,ny,nz = -x/radius, -y/radius, -z/radius
                verts.extend([x,y,z, nx,ny,nz, u,v])
    return np.asarray(verts, np.float32)


def create_crystal(height: float = 2.0, radius: float = 0.3, sides: int = 6):
    """Hexagonal crystal with pointed caps."""
    vs = []
    for i in range(sides+1):
        a = 2*math.pi*i/sides
        x,z = radius*math.cos(a), radius*math.sin(a)
        nx,nz = math.cos(a), math.sin(a)
        vs.append([x,-height/2,z, nx,0,nz, i/sides,1])
        vs.append([x, height/2,z, nx,0,nz, i/sides,0])
    vs.append([0,-height/2-radius,0, 0,-1,0, 0.5,1])
    vs.append([0, height/2+radius,0, 0, 1,0, 0.5,0])

    tris=[]
    for i in range(sides):
        a0,a1,a2,a3 = i*2, i*2+1, i*2+2, i*2+3
        tris += vs[a0]+vs[a2]+vs[a1] + vs[a1]+vs[a2]+vs[a3]
    bot, top = len(vs)-2, len(vs)-1
    for i in range(sides):
        b0,b1 = i*2, ((i+1)%sides)*2
        tris += vs[bot]+vs[b1]+vs[b0]
        t0,t1 = i*2+1, ((i+1)%sides)*2+1
        tris += vs[top]+vs[t0]+vs[t1]
    return np.asarray(tris, np.float32)


def create_water_plane(size: float = 20.0, segments: int = 50):
    verts, step = [], size/segments
    for i in range(segments+1):
        for j in range(segments+1):
            x = -size/2 + i*step
            z = -size/2 + j*step
            verts += [x,0,z, 0,1,0, i/segments, j/segments]
    tris=[]
    for i in range(segments):
        for j in range(segments):
            v0 = i*(segments+1)+j
            v1,v2,v3 = v0+1, v0+segments+1, v0+segments+2
            tris += verts[v0*8:v0*8+8]+verts[v2*8:v2*8+8]+verts[v1*8:v1*8+8]
            tris += verts[v1*8:v1*8+8]+verts[v2*8:v2*8+8]+verts[v3*8:v3*8+8]
    return np.asarray(tris, np.float32)


def create_dust(count: int = 500, bounds: float = 12.0):
    quad=[(-1,-1),(1,-1),(1,1),(-1,-1),(1,1),(-1,1)]
    vs=[]
    for _ in range(count):
        x,y,z = np.random.uniform(-bounds,bounds), np.random.uniform(-2,8), np.random.uniform(-bounds,bounds)
        sp,ph,br = np.random.uniform(0.5,2), np.random.uniform(0,2*math.pi), np.random.uniform(0.3,1)
        s=0.05
        for qx,qy in quad:
            vs += [x+qx*s,y+qy*s,z, sp,ph,br, (qx+1)*0.5,(qy+1)*0.5]
    return np.asarray(vs, np.float32)

# ----------------------------------------------------------------------------- matrix helpers
def mat_translate(x,y,z):
    return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]], np.float32)

def mat_rotate(axis, angle):
    axis = axis/np.linalg.norm(axis)
    x,y,z = axis
    c,s = math.cos(angle), math.sin(angle); t = 1-c
    return np.array([[t*x*x+c,   t*x*y-s*z, t*x*z+s*y,0],
                     [t*x*y+s*z, t*y*y+c,   t*y*z-s*x,0],
                     [t*x*z-s*y, t*y*z+s*x, t*z*z+c,  0],
                     [0,0,0,1]], np.float32)

def mat_scale(sx,sy,sz):
    return np.array([[sx,0,0,0],[0,sy,0,0],[0,0,sz,0],[0,0,0,1]], np.float32)

# ----------------------------------------------------------------------------- shader sources
# cave shaders (unchanged) -----------------------------------------------------
cave_vertex_shader = """
#version 330 core
layout(location=0) in vec3 vertex_position;
layout(location=1) in vec3 vertex_normal;
layout(location=2) in vec2 vertex_uv;
uniform mat4 u_mvp;
out vec2 fragment_uv;
void main(){gl_Position=u_mvp*vec4(vertex_position,1.0);fragment_uv=vertex_uv;}
"""
cave_fragment_shader = """
#version 330 core
in vec2 fragment_uv;
uniform sampler2D u_env;
uniform vec3 u_fog_color;uniform float u_fog_density;
vec3 fog(vec3 c,float d){float f=1-exp(-d*u_fog_density);return mix(c,u_fog_color,clamp(f,0.0,1.0));}
out vec4 FragColor;
void main(){vec3 col=texture(u_env,fragment_uv).rgb; col=fog(col,15.0); FragColor=vec4(col,1);}
"""

# crystal shaders (WITH u_mvp to satisfy Render.py) ----------------------------
crystal_vertex_shader = """
#version 330 core
layout(location=0) in vec3 vertex_position;
layout(location=1) in vec3 vertex_normal;

/* Render.py sets this every draw call */
uniform mat4 u_mvp;

/* extra matrices we manage */
uniform mat4 model_matrix;
uniform mat3 normal_matrix;

out vec3 normalWorld;
out vec3 fragPosWorld;
void main(){
    vec4 world = model_matrix*vec4(vertex_position,1.0);
    fragPosWorld = world.xyz;
    normalWorld  = normalize(normal_matrix*vertex_normal);
    gl_Position  = u_mvp*vec4(vertex_position,1.0);
}
"""

crystal_fragment_shader = """
#version 330 core
in vec3 normalWorld;
in vec3 fragPosWorld;
out vec4 FragColor;

uniform vec3  camera_position;
uniform vec3  light_direction;
uniform vec3  light_color;

uniform sampler2D u_env;

uniform vec3  crystal_base_color;
uniform float refractive_index;
uniform float roughness;
uniform float subsurface_strength;
uniform vec3  subsurface_color;
uniform vec3  emissive_color;

uniform vec3  u_fog_color;
uniform float u_fog_density;

vec3 fog(vec3 c,float d){float f=1-exp(-d*u_fog_density);return mix(c,u_fog_color,clamp(f,0.0,1.0));}
vec2 dir_to_latlong(vec3 d){
    float u=0.5+atan(d.z,d.x)/(2.0*3.14159265);
    float v=0.5-asin(clamp(d.y,-1.0,1.0))/3.14159265;
    return vec2(u,v);
}

void main(){
    vec3 N=normalize(normalWorld);
    vec3 V=normalize(camera_position-fragPosWorld);
    vec3 L=normalize(light_direction);

    /* Cook–Torrance specular */
    vec3 F0=vec3(0.04);
    float NdotL=max(dot(N,L),0.0);
    vec3 spec=vec3(0.0);
    if(NdotL>0.0){
        vec3 H=normalize(V+L);
        float NdotV=max(dot(N,V),0.0);
        float NdotH=max(dot(N,H),0.0);
        float VdotH=max(dot(V,H),0.0);
        float a=roughness*roughness, a2=a*a;
        float D=a2/(3.14159*pow((NdotH*NdotH)*(a2-1.0)+1.0,2.0));
        float k=(roughness+1.0); k=k*k*0.125;
        float G=(NdotV/(NdotV*(1.0-k)+k))*(NdotL/(NdotL*(1.0-k)+k));
        vec3 F=F0+(1.0-F0)*pow(1.0-VdotH,5.0);
        spec=light_color*D*G*F/(4.0*NdotV*NdotL+1e-6)*NdotL;
    }

    /* translucency */
    float back=max(0.0,-dot(N,L));
    vec3 trans=back*light_color*subsurface_color*subsurface_strength;

    /* env reflection / refraction */
    vec3 I=-V;
    vec3 R=reflect(I,N);
    vec3 T=refract(I,N,1.0/refractive_index);
    vec3 envR=texture(u_env,dir_to_latlong(R)).rgb;
    vec3 envT=texture(u_env,dir_to_latlong(T)).rgb;
    float cosT=max(dot(N,V),0.0);
    vec3 F=F0+(1.0-F0)*pow(1.0-cosT,5.0);
    vec3 env=envR*F+envT*(1.0-F);
    env*=crystal_base_color;

    vec3 colour=spec+trans+env+emissive_color;
    colour=fog(colour,length(fragPosWorld));
    FragColor=vec4(colour,1.0);
}
"""

# water shaders (unchanged) ----------------------------------------------------
water_vertex_shader = """
#version 330 core
layout(location=0) in vec3 vertex_position;
layout(location=1) in vec3 vertex_normal;
layout(location=2) in vec2 vertex_uv;
uniform mat4 u_mvp;uniform mat4 u_model;uniform float u_time;
out vec3 vN;out vec3 vPos;out vec2 vUV;
void main(){
    vec3 p=vertex_position;
    p.y+=sin(p.x*0.5+u_time*2.0)*0.1+sin(p.z*0.3+u_time*1.5)*0.15;
    vec4 w=u_model*vec4(p,1);gl_Position=u_mvp*vec4(p,1);
    float dx=cos(p.x*0.5+u_time*2.0)*0.05, dz=cos(p.z*0.3+u_time*1.5)*0.045;
    vN=normalize(vec3(-dx,1,-dz));vPos=w.xyz;vUV=vertex_uv;
}
"""
water_fragment_shader = """
#version 330 core
in vec3 vN;in vec3 vPos;in vec2 vUV;
uniform float u_time;uniform vec3 u_fog_color;uniform float u_fog_density;
vec3 fog(vec3 c,float d){float f=1-exp(-d*u_fog_density);return mix(c,u_fog_color,clamp(f,0.0,1.0));}
out vec4 FragColor;
void main(){
    vec3 N=normalize(vN), V=normalize(-vPos);
    float fr=pow(1.0-max(dot(N,V),0.0),3.0);
    vec3 col=mix(vec3(0.0,0.12,0.28),vec3(0.0,0.4,0.6),fr);
    float ca=sin(vUV.x*25.0+u_time)*sin(vUV.y*25.0-u_time*0.7);
    col+=vec3(smoothstep(-0.5,0.5,ca))*0.25;
    col=fog(col,length(vPos));FragColor=vec4(col,0.85);
}
"""

# dust shaders (unchanged) -----------------------------------------------------
dust_vertex_shader = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aData;
layout(location=2) in vec2 aUV;
uniform mat4 u_mvp;uniform float u_time;
out float vB;out vec2 vUV;
void main(){
    vec3 p=aPos;
    p.y+=sin(u_time*aData.x+aData.y)*0.5;
    p.x+=cos(u_time*aData.x*0.6+aData.y)*0.3;
    gl_Position=u_mvp*vec4(p,1);vB=aData.z;vUV=aUV;
}
"""
dust_fragment_shader = """
#version 330 core
in float vB;in vec2 vUV;out vec4 FragColor;
void main(){float d=distance(vUV,vec2(0.5));if(d>0.5)discard;
             float a=(1.0-smoothstep(0.2,0.5,d))*vB*0.6;
             FragColor=vec4(vec3(0.9,0.95,1.0)*vB,a);}
"""

# ----------------------------------------------------------------------------- main
def main():
    renderer = Render(lkg_size=25, debug=False)
    GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)

    stride=8*4
    attr=[(0,3,GL.GL_FLOAT,False,stride,0),
          (1,3,GL.GL_FLOAT,False,stride,12),
          (2,2,GL.GL_FLOAT,False,stride,24)]

    cave_mesh    = Mesh(create_cave(), attr)
    crystal_mesh = Mesh(create_crystal(), attr)
    water_mesh   = Mesh(create_water_plane(), attr)
    dust_mesh    = Mesh(create_dust(), attr)

    cave_shader    = Shader(cave_vertex_shader, cave_fragment_shader)
    crystal_shader = Shader(crystal_vertex_shader, crystal_fragment_shader)
    water_shader   = Shader(water_vertex_shader,  water_fragment_shader)
    dust_shader    = Shader(dust_vertex_shader,   dust_fragment_shader)

    cave_handle   = renderer.add_object(cave_mesh, cave_shader)
    water_handle  = renderer.add_object(water_mesh, water_shader)
    crystal_handles=[renderer.add_object(crystal_mesh, crystal_shader) for _ in range(60)]
    dust_handle   = renderer.add_object(dust_mesh, dust_shader)

    # ---------------- textures ------------------------------
    # If this file does not exist download it from here: https://polyhaven.com/a/small_cave
    env_tex = Texture2D(r"assets/small_cave.jpg")
    for sh in (cave_shader, crystal_shader):
        sh.use(); sh.set_uniform("u_env",0)
    env_tex.bind(0)

    # ---------------- clusters ------------------------------
    random.seed(42)
    clusters=[(random.uniform(2.5,4.5)*math.cos(a:=random.uniform(0,2*math.pi)),
               random.uniform(-1.3,-0.7),
               random.uniform(2.5,4.5)*math.sin(a)) for _ in range(5)]

    crystal_data=[]
    for i in range(60):
        base=clusters[i%5]
        rloc=random.uniform(0,0.7); aloc=random.uniform(0,2*math.pi)
        off=(rloc*math.cos(aloc), random.uniform(-0.1,0.2), rloc*math.sin(aloc))
        scale=random.uniform(0.35,0.9)
        tilt=random.uniform(0,math.radians(22)); yaw=random.uniform(0,2*math.pi)
        orient=np.array([math.sin(tilt)*math.cos(yaw), math.cos(tilt), math.sin(tilt)*math.sin(yaw)],np.float32)
        roll=random.uniform(0,2*math.pi)
        crystal_data.append((base,off,scale,orient,roll))

    # ---------------- GL state ------------------------------
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glCullFace(GL.GL_BACK); GL.glFrontFace(GL.GL_CCW)
    GL.glClearColor(0.05,0.05,0.07,1)

    fog_color,fog_density=(0.02,0.02,0.03),0.07
    start=time.time(); last=start

    cam_pos_loc=GL.glGetUniformLocation(crystal_shader.id,"camera_position")
    light_dir_loc=GL.glGetUniformLocation(crystal_shader.id,"light_direction")
    light_col_loc=GL.glGetUniformLocation(crystal_shader.id,"light_color")
    crystal_shader.use()
    GL.glUniform3f(light_dir_loc,0.2,-1.0,0.3)
    GL.glUniform3f(light_col_loc,3.0,3.0,3.0)

    while not renderer.should_close():
        now=time.time(); dt=now-last; last=now; t=now-start

        # camera matrices from Render
        view_mat, proj_mat = renderer.camera.compute_view_projection_matrices(0.5, True,
                                                                              renderer.offset,
                                                                              renderer.focus)
        cam=np.linalg.inv(view_mat)@np.array([0,0,0,1],np.float32)
        GL.glUseProgram(crystal_shader.id)
        GL.glUniform3f(cam_pos_loc,*cam[:3])

        # fog uniforms
        for sh in (cave_shader, water_shader):
            sh.use(); sh.set_uniform("u_fog_color",*fog_color); sh.set_uniform("u_fog_density",fog_density)
        water_shader.use(); water_shader.set_uniform("u_time",t)

        # static objects
        renderer.update_model(cave_handle, mat_translate(0,0,0))
        renderer.update_model(water_handle, mat_translate(0,-2.5,0))

        # crystals
        crystal_shader.use()
        for handle,(base,off,sc,orient,roll) in zip(crystal_handles, crystal_data):
            bx,by,bz=base; ox,oy,oz=off
            up=np.array([0,1,0],np.float32); axis=np.cross(up,orient)
            if np.linalg.norm(axis)<1e-5: axis=np.array([1,0,0],np.float32)
            axis/=np.linalg.norm(axis); ang=math.acos(np.clip(np.dot(up,orient),-1,1))
            model=mat_translate(bx+ox,by+oy,bz+oz) @ mat_rotate(axis,ang) @ mat_rotate(orient,roll) @ mat_scale(sc,sc*random.uniform(1.7,2.4),sc)
            normal=np.linalg.inv(model[:3,:3]).T
            crystal_shader.set_uniform_matrix("model_matrix",model)
            crystal_shader.set_uniform_matrix3("normal_matrix",normal)
            crystal_shader.set_uniform("crystal_base_color",0.8,0.85,1.0)
            crystal_shader.set_uniform("refractive_index",1.52)
            crystal_shader.set_uniform("roughness",0.18)
            crystal_shader.set_uniform("subsurface_strength",0.4)
            crystal_shader.set_uniform("subsurface_color",0.9,0.95,1.0)
            crystal_shader.set_uniform("emissive_color",0.0,0.0,0.0)
            renderer.update_model(handle, model)

        # dust
        dust_shader.use(); dust_shader.set_uniform("u_time",t)
        renderer.update_model(dust_handle, np.identity(4,np.float32))

        renderer.render_frame(dt)

if __name__=="__main__":
    main()
