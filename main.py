import taichi as ti
import math
import time

ti.init(arch=ti.vulkan)
W, H = 800, 800; MAX_NODES = 35000; N_ATTRS = 12000

attr_pos = ti.Vector.field(2, dtype=ti.f32, shape=N_ATTRS)
attr_active = ti.field(dtype=ti.i32, shape=N_ATTRS)
node_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NODES)
node_parent = ti.field(dtype=ti.i32, shape=MAX_NODES)
node_depth = ti.field(dtype=ti.i32, shape=MAX_NODES)
node_dir = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NODES)
node_pull_count = ti.field(dtype=ti.i32, shape=MAX_NODES)
node_count = ti.field(dtype=ti.i32, shape=())
trail = ti.field(dtype=ti.f32, shape=(W, H))
trail_blur = ti.field(dtype=ti.f32, shape=(W, H))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))

@ti.func
def fract(x): return x - ti.floor(x)
@ti.func
def lerp_vec(a, b, t): return a * (1.0 - t) + b * t
@ti.func
def smoothstep_val(edge0: ti.f32, edge1: ti.f32, x: ti.f32) -> ti.f32:
    t = ti.min(ti.max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)

@ti.kernel
def init_topology():
    node_count[None] = 1; node_pos[0] = ti.Vector([W * 0.5, H * 0.5]); node_parent[0] = -1; node_depth[0] = 0
    for i in range(N_ATTRS):
        th = ti.random(ti.f32) * 6.2831853; r = ti.sqrt(ti.random(ti.f32)) * (W * 0.45)
        attr_pos[i] = ti.Vector([W * 0.5 + ti.cos(th) * r, H * 0.5 + ti.sin(th) * r]); attr_active[i] = 1

@ti.kernel
def sca_grow_step(kill_dist: ti.f32, attract_dist: ti.f32, step_size: ti.f32):
    nc = node_count[None]
    for i in range(nc): node_dir[i] = ti.Vector([0.0, 0.0]); node_pull_count[i] = 0
    for i in range(N_ATTRS):
        if attr_active[i] == 1:
            ap = attr_pos[i]; min_d, min_idx = 999999.0, -1
            for j in range(node_count[None]):
                d = (ap - node_pos[j]).norm()
                if d < min_d: min_d = d; min_idx = j
            if min_d < kill_dist: attr_active[i] = 0
            elif min_d < attract_dist:
                vec = (ap - node_pos[min_idx]) / min_d
                ti.atomic_add(node_dir[min_idx][0], vec[0]); ti.atomic_add(node_dir[min_idx][1], vec[1])
                ti.atomic_add(node_pull_count[min_idx], 1)
    for i in range(nc):
        if node_pull_count[i] > 0:
            idx = ti.atomic_add(node_count[None], 1)
            if idx < MAX_NODES:
                ndir = node_dir[i].normalized()
                ang = ti.atan2(ndir.y, ndir.x) + (ti.random(ti.f32) - 0.5) * 0.2
                ndir = ti.Vector([ti.cos(ang), ti.sin(ang)])
                new_pos = node_pos[i] + ndir * step_size
                node_pos[idx] = new_pos; node_parent[idx] = i; node_depth[idx] = node_depth[i] + 1
                dist = step_size; steps = int(ti.ceil(dist * 1.5))
                thickness = ti.max(0.1, 4.0 - float(node_depth[idx]) * 0.06)
                for step in range(steps):
                    t = float(step) / float(steps); p = node_pos[i] * (1.0 - t) + new_pos * t
                    ix, iy = int(p.x), int(p.y)
                    if 0 <= ix < W and 0 <= iy < H: trail[ix, iy] += thickness * 0.8

@ti.kernel
def smooth_graph_lines():
    for i, j in trail:
        s = 0.0
        for dx, dy in ti.static([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]):
            s += trail[ti.min(ti.max(i + dx, 0), W - 1), ti.min(ti.max(j + dy, 0), H - 1)]
        trail_blur[i, j] = s / 9.0

@ti.kernel
def apply_botanical_shader():
    for i, j in pixels:
        val = ti.min(trail_blur[i, j] * 0.8, 1.0)
        dist = ti.min(ti.max((ti.Vector([float(i)/W, float(j)/H]) - ti.Vector([0.5, 0.5])).norm() * 2.0, 0.0), 1.0)
        val = ti.pow(val, 0.5 + dist * 0.4)
        col = lerp_vec(ti.Vector([0.96, 0.94, 0.90]), lerp_vec(ti.Vector([0.12, 0.08, 0.06]), ti.Vector([0.55, 0.48, 0.40]), dist), val)
        col += ti.Vector([-0.02, -0.01, 0.03]) * (smoothstep_val(0.15, 0.4, val) * smoothstep_val(0.7, 0.4, val))
        pixels[i, j] = ti.max(0.0, ti.min(1.0, col))

init_topology()
gui = ti.GUI("Scheme B: Pure Topological Vascular Graph", res=(W, H))
while gui.running:
    if node_count[None] < MAX_NODES - 100: sca_grow_step(kill_dist=6.0, attract_dist=50.0, step_size=3.5)
    smooth_graph_lines(); apply_botanical_shader(); gui.set_image(pixels); gui.show()
