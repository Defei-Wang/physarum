import taichi as ti
import numpy as np

ti.init(arch=ti.gpu, fast_math=True)
RES = 1024; N_AGENTS = 1200000; PI = 3.1415926535

trail = ti.field(dtype=ti.f32, shape=(RES, RES))
next_trail = ti.field(dtype=ti.f32, shape=(RES, RES))
scent = ti.field(dtype=ti.f32, shape=(RES, RES))
next_scent = ti.field(dtype=ti.f32, shape=(RES, RES))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))
agents = ti.Vector.field(3, dtype=ti.f32, shape=N_AGENTS)
foods = ti.Vector.field(3, dtype=ti.f32, shape=16)
time_step = ti.field(dtype=ti.f32, shape=())

SO = 22.5 * PI / 180.0; SA = 45.0 * PI / 180.0; SS = 3.5
DEP = 4.0; TRAIL_DECAY = 0.94; DIFFUSE_RATE = 0.15

@ti.kernel
def init_simulation():
    cx, cy = RES * 0.5, RES * 0.5
    for i in agents:
        cluster = ti.random(); base_x, base_y = cx, cy
        if cluster < 0.33: base_x -= 120.0; base_y -= 40.0
        elif cluster < 0.66: base_x += 120.0; base_y += 40.0
        r = ti.sqrt(ti.random()) * 45.0; theta = ti.random() * 2.0 * PI
        agents[i][0] = base_x + ti.cos(theta) * r; agents[i][1] = base_y + ti.sin(theta) * r
        agents[i][2] = theta + (ti.random() - 0.5) * 0.8

@ti.func
def sense(x: ti.f32, y: ti.f32, angle: ti.f32):
    sx = ti.cast(x + ti.cos(angle) * SS, ti.i32); sy = ti.cast(y + ti.sin(angle) * SS, ti.i32)
    val = 0.0
    if 0 < sx < RES - 1 and 0 < sy < RES - 1: val = trail[sx, sy] + scent[sx, sy] * 30.0
    return val

@ti.kernel
def step_agents():
    t = time_step[None]
    stream_dir = 1.0 if ti.sin(t * 0.15) > 0.0 else -1.0
    for i in agents:
        x, y, angle = agents[i][0], agents[i][1], agents[i][2]
        is_feeding = False
        for f in range(16):
            if foods[f][2] > 0.0:
                dx = foods[f][0] - x; dy = foods[f][1] - y
                if dx * dx + dy * dy < 64.0: is_feeding = True; foods[f][2] -= 0.005
        fwd = sense(x, y, angle); left = sense(x, y, angle - SO); right = sense(x, y, angle + SO)
        if fwd > left and fwd > right: pass
        elif fwd < left and fwd < right: angle += SA if ti.random() < 0.5 else -SA
        elif left > right: angle -= SA
        elif right > left: angle += SA
        angle += (ti.random() - 0.5) * 0.15
        flow_pulse = ti.sin(t * 2.0 - (x + y) * 0.015 * stream_dir)
        speed = 1.0; deposit = DEP
        if is_feeding: speed = 0.05; deposit = DEP * (2.5 + flow_pulse * 1.5)
        else: deposit = DEP * (0.8 + flow_pulse * 0.3)
        x += ti.cos(angle) * speed; y += ti.sin(angle) * speed
        if x < 2.0 or x >= RES - 2.0 or y < 2.0 or y >= RES - 2.0:
            x = ti.max(2.0, ti.min(RES - 2.1, x)); y = ti.max(2.0, ti.min(RES - 2.1, y)); angle = ti.random() * 2.0 * PI
        agents[i][0] = x; agents[i][1] = y; agents[i][2] = angle
        trail[ti.cast(x, ti.i32), ti.cast(y, ti.i32)] += deposit

@ti.kernel
def diffuse_and_decay():
    for i, j in trail:
        if 0 < i < RES - 1 and 0 < j < RES - 1:
            blurred = trail[i, j] * (1.0 - DIFFUSE_RATE) + (trail[i-1, j] + trail[i+1, j] + trail[i, j-1] + trail[i, j+1]) * (DIFFUSE_RATE * 0.25)
            next_trail[i, j] = blurred * TRAIL_DECAY
            s_blurred = scent[i, j] * 0.7 + (scent[i-1, j] + scent[i+1, j] + scent[i, j-1] + scent[i, j+1]) * 0.075
            next_scent[i, j] = s_blurred * 0.992

@ti.kernel
def swap_buffers():
    for i, j in trail: trail[i, j] = next_trail[i, j]; scent[i, j] = next_scent[i, j]

@ti.kernel
def render():
    for i, j in pixels:
        val = trail[i, j]; base_r, base_g, base_b = 0.05, 0.06, 0.055
        if val < 0.08: pixels[i, j] = ti.Vector([base_r, base_g, base_b])
        else:
            fringe = ti.min(1.0, val * 0.08); core = ti.pow(ti.min(1.0, val * 0.03), 3.0)
            r = base_r + fringe * 0.65 + core * 0.35; g = base_g + fringe * 0.75 + core * 0.25; b = base_b + fringe * 0.45 - core * 0.25
            pixels[i, j] = ti.Vector([ti.min(1.0, r), ti.min(1.0, g), ti.max(0.0, b)])
        for f in range(16):
            if foods[f][2] > 0.0:
                dist_sq = (foods[f][0] - i)**2 + (foods[f][1] - j)**2
                if dist_sq < 49.0:
                    alpha = ti.min(0.85, foods[f][2] / 1500.0)
                    pixels[i, j] = pixels[i, j] * (1.0 - alpha) + ti.Vector([0.78, 0.72, 0.60]) * alpha

init_simulation()
gui = ti.GUI("Physarum V1", res=(RES, RES))
food_ptr = 0
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.LMB:
            mx, my = e.pos; px, py = int(mx * RES), int(my * RES)
            foods[food_ptr % 16][0] = px; foods[food_ptr % 16][1] = py; foods[food_ptr % 16][2] = 2500.0
            scent[px, py] += 800.0; food_ptr += 1
    time_step[None] += 0.04
    step_agents(); diffuse_and_decay(); swap_buffers(); render()
    gui.set_image(pixels); gui.show()
