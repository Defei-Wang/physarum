import taichi as ti
import math

ti.init(arch=ti.vulkan)
RES = 800; N_AGENTS = 600000; N_SPECIES = 2
SENSOR_DIST = 12.0; SENSOR_ANGLE = math.pi / 7.0; TURN_ANGLE = math.pi / 5.0
STEP_SIZE = 1.2; DEPOSIT_AMT = 0.5; DECAY_FACTOR = 0.93; BLUR_PASSES = 2

attr_matrix = ti.math.mat2([[1.0, -0.4], [-0.4, 1.0]])
grid = ti.Vector.field(N_SPECIES, dtype=ti.f32, shape=(RES, RES))
grid_new = ti.Vector.field(N_SPECIES, dtype=ti.f32, shape=(RES, RES))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))
pos = ti.Vector.field(2, dtype=ti.f32, shape=N_AGENTS)
heading = ti.field(dtype=ti.f32, shape=N_AGENTS)
species = ti.field(dtype=ti.i32, shape=N_AGENTS)

@ti.func
def fract_val(x: ti.f32) -> ti.f32: return x - ti.floor(x)
@ti.func
def fract_vec(v): return ti.Vector([fract_val(v.x), fract_val(v.y)])
@ti.func
def hash_rand(coord):
    SQ2 = 1414.21356; dist = ti.sqrt((coord.x - 0.1618)**2 + (coord.y - 0.314)**2)
    return fract_val(ti.tan(dist) * SQ2)
@ti.func
def lerp_vec(a, b, t): return a * (1.0 - t) + b * t

@ti.func
def sample_field_bilinear(x: ti.f32, y: ti.f32, sp: ti.i32) -> ti.f32:
    x = fract_val(x / RES) * RES; y = fract_val(y / RES) * RES
    x0, y0 = int(x), int(y); x1, y1 = (x0 + 1) % RES, (y0 + 1) % RES
    fx, fy = x - x0, y - y0
    w0 = attr_matrix[sp, 0]; w1 = attr_matrix[sp, 1]
    v00 = grid[x0, y0][0] * w0 + grid[x0, y0][1] * w1
    v10 = grid[x1, y0][0] * w0 + grid[x1, y0][1] * w1
    v01 = grid[x0, y1][0] * w0 + grid[x0, y1][1] * w1
    v11 = grid[x1, y1][0] * w0 + grid[x1, y1][1] * w1
    return v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy

@ti.kernel
def init_simulation():
    for i in range(N_AGENTS):
        pos[i] = ti.Vector([ti.random(ti.f32) * RES, ti.random(ti.f32) * RES])
        heading[i] = ti.random(ti.f32) * math.pi * 2.0
        species[i] = 0 if ti.random(ti.f32) < 0.5 else 1
    for i, j in grid: grid[i, j] = ti.Vector([0.0, 0.0])

@ti.kernel
def step_agents():
    for i in range(N_AGENTS):
        p = pos[i]; ang = heading[i]; sp = species[i]
        dir_f = ti.Vector([ti.cos(ang), ti.sin(ang)])
        dir_l = ti.Vector([ti.cos(ang - SENSOR_ANGLE), ti.sin(ang - SENSOR_ANGLE)])
        dir_r = ti.Vector([ti.cos(ang + SENSOR_ANGLE), ti.sin(ang + SENSOR_ANGLE)])
        v_f = sample_field_bilinear(p.x + dir_f.x * SENSOR_DIST, p.y + dir_f.y * SENSOR_DIST, sp)
        v_l = sample_field_bilinear(p.x + dir_l.x * SENSOR_DIST, p.y + dir_l.y * SENSOR_DIST, sp)
        v_r = sample_field_bilinear(p.x + dir_r.x * SENSOR_DIST, p.y + dir_r.y * SENSOR_DIST, sp)
        if v_f > v_l and v_f > v_r: pass
        elif v_f < v_l and v_f < v_r: ang += TURN_ANGLE if hash_rand(p) < 0.5 else -TURN_ANGLE
        elif v_l > v_r: ang -= TURN_ANGLE
        elif v_r > v_l: ang += TURN_ANGLE
        p += ti.Vector([ti.cos(ang), ti.sin(ang)]) * STEP_SIZE
        p = fract_vec(p / RES) * RES
        pos[i] = p; heading[i] = ang
        ix, iy = int(p.x), int(p.y)
        if 0 <= ix < RES and 0 <= iy < RES: ti.atomic_add(grid[ix, iy][sp], DEPOSIT_AMT)

@ti.kernel
def diffuse_and_decay(is_final_pass: ti.i32):
    for i, j in grid:
        s0, s1 = 0.0, 0.0
        for dx, dy in ti.static([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]):
            ni, nj = (i + dx + RES) % RES, (j + dy + RES) % RES
            s0 += grid[ni, nj][0]; s1 += grid[ni, nj][1]
        v0, v1 = s0 / 9.0, s1 / 9.0
        if is_final_pass == 1: v0 *= DECAY_FACTOR; v1 *= DECAY_FACTOR
        grid_new[i, j] = ti.Vector([v0, v1])
    for i, j in grid: grid[i, j] = grid_new[i, j]

@ti.kernel
def interact_mouse(mx: ti.f32, my: ti.f32):
    px, py = mx * RES, my * RES
    for i in range(N_AGENTS):
        if (pos[i] - ti.Vector([px, py])).norm() < 30.0:
            heading[i] += math.pi; grid[int(pos[i].x), int(pos[i].y)][species[i]] += 20.0

@ti.kernel
def render_2_5d_normal():
    for i, j in pixels:
        h_c = grid[i, j][0] + grid[i, j][1]
        h_l = grid[(i - 1 + RES) % RES, j][0] + grid[(i - 1 + RES) % RES, j][1]
        h_r = grid[(i + 1 + RES) % RES, j][0] + grid[(i + 1 + RES) % RES, j][1]
        h_d = grid[i, (j - 1 + RES) % RES][0] + grid[i, (j - 1 + RES) % RES][1]
        h_u = grid[i, (j + 1 + RES) % RES][0] + grid[i, (j + 1 + RES) % RES][1]
        dx = h_r - h_l; dy = h_u - h_d
        bump_scale = 0.15; N = ti.Vector([-dx * bump_scale, -dy * bump_scale, 1.0]).normalized()
        L = ti.Vector([0.7, 0.7, 0.8]).normalized(); diffuse = ti.max(0.0, N.dot(L))
        bg = ti.Vector([0.05, 0.05, 0.06]); tissue = ti.Vector([0.85, 0.88, 0.90])
        mask = ti.min(h_c * 0.1, 1.0)
        base_color = lerp_vec(bg, tissue, mask); lit_color = base_color * (0.15 + diffuse * 0.9)
        pixels[i, j] = ti.max(0.0, ti.min(1.0, lit_color))

init_simulation()
gui = ti.GUI("The Ultimate 2.5D Physarum", res=(RES, RES))
while gui.running:
    if gui.is_pressed(ti.GUI.LMB): mx, my = gui.get_cursor_pos(); interact_mouse(mx, my)
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == 'r': init_simulation()
    step_agents()
    for p in range(BLUR_PASSES):
        is_final = 1 if p == BLUR_PASSES - 1 else 0
        diffuse_and_decay(is_final)
    render_2_5d_normal()
    gui.set_image(pixels); gui.show()
