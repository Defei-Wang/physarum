# 遇到严重错误立即停止，杜绝雪崩报错
$ErrorActionPreference = "Stop"

$exportPath = "C:\Users\Administrator\Documents\physarum_release"

# 1. 创建绝对路径的干净发布目录
if (Test-Path $exportPath) {
    Remove-Item -Recurse -Force $exportPath
}
New-Item -ItemType Directory -Force -Path $exportPath | Out-Null
Set-Location -LiteralPath $exportPath

# 2. 初始化 Git 仓库与本地环境保底
git init
git config core.autocrlf false

# 避免未配置 Git 用户信息导致 commit 阻断
$gitUser = git config user.name
if (-not $gitUser) {
    git config user.name "Difei Wang"
    git config user.email "difei.wang@outlook.com"
}

@"
__pycache__/
*.pyc
.vscode/
"@ | Out-File -FilePath ".gitignore" -Encoding utf8

# 3. 写入中英文双语 README.md
$readmeContent = @"
# Physarum Polycephalum Morphogenesis

English | [中文](#中文介绍)

A collection of high-performance GPU-accelerated simulations exploring the morphogenesis and transport network formation of *Physarum polycephalum* (slime mold). Implemented in Python using the Taichi rendering and compute framework.

This repository chronicles the evolution of the simulation architecture from simple multi-agent systems to complex, multi-species fluid-graph hybrids with 2.5D normal shading.

## Version History (Tags)

*   **[v1.0] Baseline Dense Swarm**: Initial 1.2M dense agent swarm with shuttle streaming and basic 3-point sensing.
*   **[v4.0-ianpilon] Generational Tip-Growth**: Explicit branching, finite agent lifespan, and botanical archival shading (Scheme A).
*   **[v5.0-sca-graph] Space Colonization Graph**: Pure topological vascular graph using discrete attractors to completely eliminate sponge-like artifacts (Scheme B).
*   **[v6.0-nicoptere] Dual-Channel Blending**: 1:1 translation of the classic WebGL implementation featuring dual-channel R/G delayed blending and torus topology.
*   **[v7.0-fogleman-normal] Multi-Species & 2.5D Normal**: The ultimate morphogenesis engine. Features a multi-species attraction matrix, multi-pass blur, and 2.5D height-field normal bump mapping for an electron-microscope aesthetic.

---

<h2 id="中文介绍">中文介绍</h2>

基于 Python 和 Taichi (Vulkan/CUDA) 框架开发的多头绒泡菌（黏菌）形态发生与传输管网高真实感物理模拟实验库。

本项目记录了黏菌算法架构的演进过程：从最基础的大规模纯粒子游走，到代际出芽分叉、空间殖民拓扑图，再到引入多物种拮抗矩阵与 2.5D 法线立体渲染的终极混合管线。

## 版本演进 (Git Tags)

*   **[v1.0] 基础高密粒子流**: 120万粒子规模的基础模型，带有穿梭流控制器与双尺度非线性显色。
*   **[v4.0-ianpilon] 代际分叉与顶端生长**: 引入显式递归出芽、代际寿命递减（Gen 0\~6）与古典植物标本着色（方案 A）。
*   **[v5.0-sca-graph] 空间殖民图网络**: 彻底脱离网格连续场，采用离散养分点与拓扑图算法，从数学上根除海绵化（方案 B）。
*   **[v6.0-nicoptere] 双通道延迟混合**: 1:1 复刻经典 WebGL 实现，引入 R/G 双通道延迟混合与无缝甜甜圈拓扑，彻底消除锯齿与打转。
*   **[v7.0-fogleman-normal] 多物种与 2.5D 立体渲染**: 终极视觉形态。采用多物种交叉吸引矩阵撕裂张力网，并通过空间差分求梯度，实现 2.5D 高度场法线光照（电子显微镜质感）。
"@
$readmeContent | Out-File -FilePath "README.md" -Encoding utf8
git add .gitignore README.md
git commit -m "docs: add bilingual README and gitignore"

# 4. 写入 V1 并打标
$v1Code = @"
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
"@
$v1Code | Out-File -FilePath "main.py" -Encoding utf8
git add main.py
git commit -m "feat(v1): initial 1.2M dense agent swarm with shuttle streaming"
git tag v1.0

# 5. 写入 V4 并打标
$v4Code = @"
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
"@
$v4Code | Out-File -FilePath "main.py" -Encoding utf8
git add main.py
git commit -m "feat(v4): introduce generational tip-growth and botanical archival shader"
git tag v4.0-ianpilon

# 6. 写入 V5 并打标
$v5Code = @"
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
"@
$v5Code | Out-File -FilePath "main.py" -Encoding utf8
git add main.py
git commit -m "feat(v5): pure space colonization graph with discrete attractors"
git tag v5.0-sca-graph

# 7. 写入 V6 并打标
$v6Code = @"
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
"@
$v6Code | Out-File -FilePath "main.py" -Encoding utf8
git add main.py
git commit -m "feat(v6): 1:1 nicoptere dual-channel blending on torus topology"
git tag v6.0-nicoptere

# 8. 写入 V7 并打标
$v7Code = @"
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
"@
$v7Code | Out-File -FilePath "main.py" -Encoding utf8
git add main.py
git commit -m "feat(v7): multi-species attraction matrix and 2.5D normal bump mapping"
git tag v7.0-fogleman-normal

# 9. 自动通过 GitHub CLI 建立仓库并推送到远端
Write-Host "Creating GitHub repository 'physarum' and pushing all branches and tags..." -ForegroundColor Green
& "C:\Program Files\GitHub CLI\gh.exe" repo create physarum --public --source=. --remote=origin --push
git push origin --tags

Write-Host "Deployment completed successfully! Repository published at 'physarum'." -ForegroundColor Cyan