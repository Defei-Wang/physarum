import taichi as ti
import math
import time

ti.init(arch=ti.vulkan)
N_AGENTS = 512 * 512; RES = 800
field_R = ti.field(dtype=ti.f32, shape=(RES, RES))
field_G = ti.field(dtype=ti.f32, shape=(RES, RES))
field_G_new = ti.field(dtype=ti.f32, shape=(RES, RES))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))
pos = ti.Vector.field(2, dtype=ti.f32, shape=N_AGENTS)
angle = ti.field(dtype=ti.f32, shape=N_AGENTS)

DECAY = 0.90; SA_RAW = 2.0; RA_RAW = 4.0; SO_RAW = 12.0; SS_RAW = 1.1
RAD = 1.0 / math.pi; SA = SA_RAW * RAD; RA = RA_RAW * RAD

@ti.func
def fract_val(x: ti.f32) -> ti.f32: return x - ti.floor(x)
@ti.func
def fract_vec(v): return ti.Vector([fract_val(v.x), fract_val(v.y)])
@ti.func
def hash_rand(coord, time_val: ti.f32):
    SQ2 = 1.41421356237 * 1000.0; PHI = 1.61803398875 * 0.1
    c2 = coord * (time_val + PHI); dist = ti.sqrt((c2.x - PHI) ** 2 + (c2.y - math.pi * 0.1) ** 2)
    return fract_val(ti.tan(dist) * SQ2)

@ti.kernel
def init_simulation():
    for i in range(N_AGENTS): pos[i] = ti.Vector([ti.random(ti.f32), ti.random(ti.f32)]); angle[i] = ti.random(ti.f32) * math.pi * 2.0

@ti.func
def get_trail_value(uv) -> ti.f32:
    wrapped_uv = fract_vec(uv); ix = int(wrapped_uv.x * RES); iy = int(wrapped_uv.y * RES)
    return field_G[ix, iy]

@ti.kernel
def update_agents_and_deposit(time_val: ti.f32):
    for i, j in field_R: field_R[i, j] = 0.0
    for i in range(N_AGENTS):
        p = pos[i]; ang = angle[i]
        so_vec = SO_RAW / RES; ss_vec = SS_RAW / RES
        uvFL = p + ti.Vector([ti.cos(ang - SA), ti.sin(ang - SA)]) * so_vec
        uvF  = p + ti.Vector([ti.cos(ang), ti.sin(ang)]) * so_vec
        uvFR = p + ti.Vector([ti.cos(ang + SA), ti.sin(ang + SA)]) * so_vec
        FL = get_trail_value(uvFL); F  = get_trail_value(uvF); FR = get_trail_value(uvFR)
        if F > FL and F > FR: pass
        elif F < FL and F < FR:
            if hash_rand(p, time_val) > 0.5: ang += RA
            else: ang -= RA
        elif FL < FR: ang += RA
        elif FL > FR: ang -= RA
        p += ti.Vector([ti.cos(ang), ti.sin(ang)]) * ss_vec; p = fract_vec(p)
        pos[i] = p; angle[i] = ang
        ix = int(p.x * RES); iy = int(p.y * RES)
        if 0 <= ix < RES and 0 <= iy < RES: field_R[ix, iy] += 1.0

@ti.kernel
def diffuse_and_decay():
    weight = 1.0 / 9.0
    for i, j in field_G:
        col = 0.0
        for dx, dy in ti.static([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]):
            ni = (i + dx + RES) % RES; nj = (j + dy + RES) % RES
            val_r = field_R[ni, nj]; val_g = field_G[ni, nj]
            col += val_r * weight + val_g * weight * 0.5
        field_G_new[i, j] = col * DECAY
    for i, j in field_G: field_G[i, j] = field_G_new[i, j]

@ti.kernel
def interact_mouse(mx: ti.f32, my: ti.f32):
    count = 1200; radius = 0.03; u, v = mx, my
    for i in range(count):
        idx = int(ti.random(ti.f32) * N_AGENTS); a = ti.random(ti.f32) * math.pi * 2.0; r = ti.random(ti.f32) * radius
        pos[idx] = fract_vec(ti.Vector([u + ti.cos(a) * r, v + ti.sin(a) * r]))

@ti.kernel
def render_postprocess():
    for i, j in pixels:
        val = field_G[i, j]; val = ti.min(ti.max(val, 0.0), 1.0)
        pixels[i, j] = ti.Vector([val, val, val])

init_simulation()
gui = ti.GUI("1:1 nicoptere/physarum Translation", res=(RES, RES))
start_t = time.time()
while gui.running:
    if gui.is_pressed(ti.GUI.LMB):
        mx, my = gui.get_cursor_pos(); interact_mouse(mx, my)
    cur_time = float(time.time() - start_t)
    update_agents_and_deposit(cur_time); diffuse_and_decay(); render_postprocess()
    gui.set_image(pixels); gui.show()
