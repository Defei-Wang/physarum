import taichi as ti
import math
import time

ti.init(arch=ti.vulkan)
W, H = 800, 800; MAX_AGENTS = 30000

trail = ti.field(dtype=ti.f32, shape=(W, H))
trail_new = ti.field(dtype=ti.f32, shape=(W, H))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_AGENTS)
angle = ti.field(dtype=ti.f32, shape=MAX_AGENTS)
active = ti.field(dtype=ti.i32, shape=MAX_AGENTS)
gen = ti.field(dtype=ti.i32, shape=MAX_AGENTS)
life = ti.field(dtype=ti.f32, shape=MAX_AGENTS)
max_life = ti.field(dtype=ti.f32, shape=MAX_AGENTS)
deposit_strength = ti.field(dtype=ti.f32, shape=MAX_AGENTS)
alloc_counter = ti.field(dtype=ti.i32, shape=())

@ti.func
def fract(x): return x - ti.floor(x)
@ti.func
def lerp_vec(a, b, t): return a * (1.0 - t) + b * t
@ti.func
def lerp_val(a: ti.f32, b: ti.f32, t: ti.f32) -> ti.f32: return a * (1.0 - t) + b * t
@ti.func
def smoothstep_val(edge0: ti.f32, edge1: ti.f32, x: ti.f32) -> ti.f32:
    t = ti.min(ti.max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)
@ti.func
def hash21(p): return fract(ti.sin(p.dot(ti.Vector([127.1, 311.7]))) * 43758.5453)
@ti.func
def noise2d(p):
    i = ti.floor(p); f = fract(p); u = f * f * (3.0 - 2.0 * f)
    return lerp_val(lerp_val(hash21(i), hash21(i + ti.Vector([1.0, 0.0])), u.x),
                    lerp_val(hash21(i + ti.Vector([0.0, 1.0])), hash21(i + ti.Vector([1.0, 1.0])), u.x), u.y)
@ti.func
def fbm(p):
    v = 0.0; a = 0.5
    for _ in ti.static(range(4)):
        v += a * noise2d(p); p *= 2.0; a *= 0.5
    return v
@ti.func
def sample_trail(p, ang, dist, w, h):
    sx = int(p.x + ti.cos(ang) * dist); sy = int(p.y + ti.sin(ang) * dist)
    return trail[sx, sy] if 0 <= sx < w and 0 <= sy < h else 0.0

@ti.kernel
def spawn_batch(cx: ti.f32, cy: ti.f32, count: ti.i32, is_disc: ti.i32, core_r: ti.f32, base_life: ti.f32):
    for _ in range(count):
        idx = ti.atomic_add(alloc_counter[None], 1) % MAX_AGENTS
        active[idx], gen[idx], deposit_strength[idx] = 1, 0, 1.0
        ml = base_life * (0.8 + ti.random(ti.f32) * 0.4)
        max_life[idx] = life[idx] = ml
        if is_disc == 1:
            r = ti.sqrt(ti.random(ti.f32)) * core_r; th = ti.random(ti.f32) * 6.2831853
            pos[idx] = ti.Vector([cx + ti.cos(th) * r, cy + ti.sin(th) * r])
            angle[idx] = th + (ti.random(ti.f32) - 0.5) * 0.8
        else:
            pos[idx] = ti.Vector([cx + (ti.random(ti.f32) - 0.5) * 8.0, cy + (ti.random(ti.f32) - 0.5) * 8.0])
            angle[idx] = ti.random(ti.f32) * 6.2831853

@ti.kernel
def diffuse_and_decay(decay: ti.f32):
    for i, j in trail:
        s = 0.0
        for dx, dy in ti.static([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]):
            s += trail[ti.min(ti.max(i + dx, 0), W - 1), ti.min(ti.max(j + dy, 0), H - 1)]
        val = (s / 9.0) * decay
        trail_new[i, j] = 0.0 if val < 0.003 else val

@ti.kernel
def update_agents(s_ang: ti.f32, s_dst: ti.f32, t_spd: ti.f32, m_spd: ti.f32, f_ch: ti.f32, f_ang: ti.f32, f_acc: ti.f32, max_g: ti.i32, v_tap: ti.f32, l_tap: ti.f32, b_life: ti.f32):
    for i in range(MAX_AGENTS):
        if active[i] == 1:
            life[i] -= 1.0
            if life[i] <= 0.0: active[i] = 0
            else:
                p, ang, g = pos[i], angle[i], gen[i]
                vl, vc, vr = sample_trail(p, ang - s_ang, s_dst, W, H), sample_trail(p, ang, s_dst, W, H), sample_trail(p, ang + s_ang, s_dst, W, H)
                if vc > vl and vc > vr: pass
                elif vc < vl and vc < vr: ang += (ti.random(ti.f32) - 0.5) * 2.0 * t_spd
                elif vl > vr: ang -= t_spd
                elif vr > vl: ang += t_spd
                cur_f_ch = f_ch * ti.pow(f_acc, float(g))
                if g < max_g and ti.random(ti.f32) < cur_f_ch:
                    c_idx = ti.atomic_add(alloc_counter[None], 1) % MAX_AGENTS
                    active[c_idx], pos[c_idx], gen[c_idx] = 1, p, g + 1
                    deposit_strength[c_idx] = deposit_strength[i] * v_tap
                    sign = 1.0 if ti.random(ti.f32) > 0.5 else -1.0
                    defl = f_ang * (0.5 + ti.random(ti.f32) * 0.5)
                    angle[c_idx] = ang + sign * defl
                    c_ml = b_life * ti.pow(l_tap, float(g + 1)) * (0.8 + ti.random(ti.f32) * 0.4)
                    max_life[c_idx] = life[c_idx] = c_ml
                    ang -= sign * defl * 0.15
                ang += (ti.random(ti.f32) - 0.5) * (0.08 + float(g) * 0.04)
                p += ti.Vector([ti.cos(ang), ti.sin(ang)]) * ti.max(0.3, m_spd * (1.0 - float(g) * 0.05))
                if p.x < 0 or p.x >= W or p.y < 0 or p.y >= H: active[i] = 0
                else:
                    pos[i], angle[i] = p, ang
                    trail[int(p.x), int(p.y)] = ti.min(1.0, trail[int(p.x), int(p.y)] + deposit_strength[i] * (life[i] / max_life[i]))

@ti.kernel
def render_botanical(u_time: ti.f32):
    for i, j in pixels:
        val = trail[i, j]
        dist = ti.min(ti.max((ti.Vector([float(i)/W, float(j)/H]) - ti.Vector([0.5, 0.5])).norm() * 2.0, 0.0), 1.0)
        val = ti.pow(val, 0.5 + dist * 0.4)
        col = lerp_vec(ti.Vector([0.96, 0.94, 0.90]), lerp_vec(ti.Vector([0.12, 0.08, 0.06]), ti.Vector([0.55, 0.48, 0.40]), dist), val)
        col += ti.Vector([-0.02, -0.01, 0.03]) * (smoothstep_val(0.15, 0.4, val) * smoothstep_val(0.7, 0.4, val))
        p_noise = smoothstep_val(0.35, 0.65, fbm(ti.Vector([float(i), float(j)]) * 0.04 + u_time * 0.05))
        col = lerp_vec(col, ti.Vector([0.96, 0.94, 0.90]), p_noise * (smoothstep_val(0.05, 0.25, val) * smoothstep_val(0.65, 0.35, val)) * 0.5 * 0.6)
        grain = (hash21(ti.Vector([float(i), float(j)]) * 1.1 + u_time) - 0.5) * 0.08 + (hash21(ti.Vector([float(i), float(j)]) * 3.7 + u_time * 0.7) - 0.5) * 0.04 * val
        pixels[i, j] = ti.max(0.0, ti.min(1.0, col + ti.Vector([grain, grain, grain])))

gui = ti.GUI("Scheme A: Generational Tip Flow", res=(W, H))
spawn_batch(W * 0.5, H * 0.5, 100, 1, 50.0, 400.0)
start_t = time.time()
while gui.running:
    if gui.get_event(ti.GUI.LMB):
        mx, my = gui.get_cursor_pos()
        spawn_batch(mx * W, my * H, 40, 0, 8.0, 400.0)
    diffuse_and_decay(0.985); trail.copy_from(trail_new)
    spawn_batch(W * 0.5, H * 0.5, 120, 1, 50.0, 400.0)
    update_agents(0.5, 20.0, 0.2, 1.0, 0.012, 0.78, 1.3, 6, 0.55, 0.6, 400.0)
    render_botanical(float(time.time() - start_t))
    gui.set_image(pixels); gui.show()
