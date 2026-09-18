import math
import time
import taichi as ti

# =========================================================================
# 硬件与系统架构配置 (目标: RTX 3050 6GB / CUDA / GGUI 60FPS)
# =========================================================================
ti.init(arch=ti.cuda, fast_math=True)

RES = 900                           # 视口与网格分辨率
MAX_NODES = 45000                   # 拓扑图节点容量上限
MAX_ATTRS = 28000                   # 养分引力子上限
MAX_FOODS = 64                      # 预先布置的站点数量上限
PI = 3.141592653589793

# 空间殖民与仿生形态参数
KILL_DIST = 7.5                     # 养分吸收消亡半径
ATTRACT_DIST = 38.0                 # 局部嗅探视野半径 (无全局天眼)
STEP_SIZE = 3.2                     # 单步出芽延伸长度
BROWNIAN_NOISE = 0.22               # 仿生蜿蜒扰动弧度
MAX_BRANCH_PER_NODE = 3             # 单节点出芽分支数上限

# GPU 空间哈希加速桶
CELL_SIZE = 40.0
GRID_W = int(math.ceil(RES / CELL_SIZE))
GRID_H = int(math.ceil(RES / CELL_SIZE))
MAX_NODES_PER_CELL = 128

# 连续组织场、水力代谢与凋亡参数
DISH_RADIUS = RES * 0.465           # 圆形培养皿刚性物理边界
TRAIL_DECAY = 0.88                  # 连续场物理挥发率
FLOW_DECAY = 0.955                  # 输运通量半衰期
VITALITY_DECAY_RATE = 0.0035        # 盲端能耗惩罚与枯萎速率
NORMAL_BUMP = 0.36                  # 2.5D 微观表面法线强度

# =========================================================================
# 显存字段显式分配
# =========================================================================
node_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NODES)
node_parent = ti.field(dtype=ti.i32, shape=MAX_NODES)
node_depth = ti.field(dtype=ti.i32, shape=MAX_NODES)
node_flow = ti.field(dtype=ti.f32, shape=MAX_NODES)
node_vitality = ti.field(dtype=ti.f32, shape=MAX_NODES)    # 生物存活活力值 (0.0\~1.0)
node_active = ti.field(dtype=ti.i32, shape=MAX_NODES)      # 存活状态掩码
node_children = ti.field(dtype=ti.i32, shape=MAX_NODES)
node_dir = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NODES)
node_pull_count = ti.field(dtype=ti.i32, shape=MAX_NODES)
node_count = ti.field(dtype=ti.i32, shape=())

attr_pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_ATTRS)
attr_active = ti.field(dtype=ti.i32, shape=MAX_ATTRS)
attr_cursor = ti.field(dtype=ti.i32, shape=())

cell_count = ti.field(dtype=ti.i32, shape=(GRID_W, GRID_H))
cell_nodes = ti.field(dtype=ti.i32, shape=(GRID_W, GRID_H, MAX_NODES_PER_CELL))

# 食物数据表: [x, y, 剩余能量, 存活状态]
foods = ti.Vector.field(4, dtype=ti.f32, shape=MAX_FOODS)
food_eating_flag = ti.field(dtype=ti.i32, shape=MAX_FOODS)

# 仿真状态标量
is_inoculated = ti.field(dtype=ti.i32, shape=())           # 是否已接种黏菌母核

trail = ti.field(dtype=ti.f32, shape=(RES, RES))
trail_blur = ti.field(dtype=ti.f32, shape=(RES, RES))
render_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))


# =========================================================================
# 核心仿真内核函数
# =========================================================================
@ti.kernel
def reset_simulation():
    is_inoculated[None] = 0
    node_count[None] = 0
    attr_cursor[None] = 0

    for i in range(MAX_NODES):
        node_parent[i] = -1
        node_flow[i] = 0.0
        node_vitality[i] = 0.0
        node_active[i] = 0
        node_children[i] = 0

    for i in range(MAX_ATTRS):
        attr_active[i] = 0

    for i in range(MAX_FOODS):
        foods[i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
        food_eating_flag[i] = 0

    for i, j in trail:
        trail[i, j] = 0.0
        trail_blur[i, j] = 0.0
        render_buffer[i, j] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def add_food_site(px: ti.f32, py: ti.f32, energy: ti.f32):
    """仅向网格布设食物与局域养分囊，严禁生成全局指向的直连引力线"""
    target_slot = -1
    for f in range(MAX_FOODS):
        if foods[f][3] < 0.5 and target_slot == -1:
            target_slot = f

    if target_slot != -1:
        foods[target_slot] = ti.Vector([px, py, energy, 1.0])

        # 仅在食物局域微环境散布高浓度引力点 (无全局天眼)
        for _ in range(320):
            idx = ti.atomic_add(attr_cursor[None], 1) % MAX_ATTRS
            r = ti.sqrt(ti.random(ti.f32)) * 36.0
            th = ti.random(ti.f32) * 2.0 * PI
            attr_pos[idx] = ti.Vector([px + ti.cos(th) * r, py + ti.sin(th) * r])
            attr_active[idx] = 1


@ti.kernel
def inoculate_slime_mold(px: ti.f32, py: ti.f32):
    """右键接种黏菌母核，并向底质释放全盘均匀的探索性引力子"""
    center = ti.Vector([px, py])
    node_count[None] = 1
    node_pos[0] = center
    node_parent[0] = -1
    node_depth[0] = 0
    node_flow[0] = 2.0
    node_vitality[0] = 1.0
    node_active[0] = 1
    node_children[0] = 0

    # 散布各向同性基质引力子，驱动全域盲目出芽探索
    dish_center = ti.Vector([ti.cast(RES, ti.f32) * 0.5, ti.cast(RES, ti.f32) * 0.5])
    for _ in range(12000):
        idx = ti.atomic_add(attr_cursor[None], 1) % MAX_ATTRS
        th = ti.random(ti.f32) * 2.0 * PI
        r = ti.sqrt(ti.random(ti.f32)) * (DISH_RADIUS - 10.0)
        attr_pos[idx] = dish_center + ti.Vector([ti.cos(th), ti.sin(th)]) * r
        attr_active[idx] = 1

    is_inoculated[None] = 1


@ti.kernel
def bin_nodes_to_grid():
    for gx, gy in cell_count:
        cell_count[gx, gy] = 0

    nc = node_count[None]
    for i in range(nc):
        if node_active[i] == 1:
            gx = ti.cast(ti.floor(node_pos[i].x / CELL_SIZE), ti.i32)
            gy = ti.cast(ti.floor(node_pos[i].y / CELL_SIZE), ti.i32)
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                idx = ti.atomic_add(cell_count[gx, gy], 1)
                if idx < MAX_NODES_PER_CELL:
                    cell_nodes[gx, gy, idx] = i


@ti.kernel
def sca_search_and_pull():
    nc = node_count[None]
    for i in range(nc):
        node_dir[i] = ti.Vector([0.0, 0.0])
        node_pull_count[i] = 0

    for i in range(MAX_ATTRS):
        if attr_active[i] == 1:
            ap = attr_pos[i]
            gx = ti.cast(ti.floor(ap.x / CELL_SIZE), ti.i32)
            gy = ti.cast(ti.floor(ap.y / CELL_SIZE), ti.i32)

            min_d = ATTRACT_DIST
            min_idx = -1

            for dx, dy in ti.static([(-1, -1), (-1, 0), (-1, 1),
                                     (0, -1),  (0, 0),  (0, 1),
                                     (1, -1),  (1, 0),  (1, 1)]):
                cx, cy = gx + dx, gy + dy
                if 0 <= cx < GRID_W and 0 <= cy < GRID_H:
                    cnt = cell_count[cx, cy]
                    lim = ti.min(cnt, MAX_NODES_PER_CELL)
                    for k in range(lim):
                        n_idx = cell_nodes[cx, cy, k]
                        if node_active[n_idx] == 1:
                            d = (ap - node_pos[n_idx]).norm()
                            if d < min_d:
                                min_d = d
                                min_idx = n_idx

            if min_d < KILL_DIST:
                attr_active[i] = 0
            elif min_idx != -1:
                vec = (ap - node_pos[min_idx]) / (min_d + 1e-4)
                ti.atomic_add(node_dir[min_idx][0], vec[0])
                ti.atomic_add(node_dir[min_idx][1], vec[1])
                ti.atomic_add(node_pull_count[min_idx], 1)


@ti.kernel
def sca_grow_step():
    nc = node_count[None]
    center = ti.Vector([ti.cast(RES, ti.f32) * 0.5, ti.cast(RES, ti.f32) * 0.5])

    for i in range(nc):
        if node_active[i] == 1 and node_pull_count[i] > 0 and node_children[i] < MAX_BRANCH_PER_NODE:
            idx = ti.atomic_add(node_count[None], 1)
            if idx < MAX_NODES:
                ndir = node_dir[i].normalized()
                ang = ti.atan2(ndir.y, ndir.x) + (ti.random(ti.f32) - 0.5) * BROWNIAN_NOISE
                growth_vec = ti.Vector([ti.cos(ang), ti.sin(ang)]) * STEP_SIZE
                new_pos = node_pos[i] + growth_vec

                if (new_pos - center).norm() < (DISH_RADIUS - 3.0):
                    node_pos[idx] = new_pos
                    node_parent[idx] = i
                    node_depth[idx] = node_depth[i] + 1
                    node_flow[idx] = 0.12
                    node_vitality[idx] = 1.0
                    node_active[idx] = 1
                    node_children[idx] = 0
                    node_children[i] += 1
                else:
                    node_count[None] = idx


@ti.kernel
def retrograde_flow_and_prune():
    """局部触碰进食、逆行水力干道粗化与盲端能量代谢剪枝"""
    for f in range(MAX_FOODS):
        food_eating_flag[f] = 0
        if foods[f][3] > 0.5:
            fp = foods[f][:2]
            gx = ti.cast(ti.floor(fp.x / CELL_SIZE), ti.i32)
            gy = ti.cast(ti.floor(fp.y / CELL_SIZE), ti.i32)

            nearest_node = -1
            min_d = 22.0

            for dx, dy in ti.static([(-1, -1), (-1, 0), (-1, 1),
                                     (0, -1),  (0, 0),  (0, 1),
                                     (1, -1),  (1, 0),  (1, 1)]):
                cx, cy = gx + dx, gy + dy
                if 0 <= cx < GRID_W and 0 <= cy < GRID_H:
                    lim = ti.min(cell_count[cx, cy], MAX_NODES_PER_CELL)
                    for k in range(lim):
                        n_idx = cell_nodes[cx, cy, k]
                        if node_active[n_idx] == 1:
                            d = (fp - node_pos[n_idx]).norm()
                            if d < min_d:
                                min_d = d
                                nearest_node = n_idx

            if nearest_node != -1:
                food_eating_flag[f] = 1
                curr = nearest_node
                steps = 0
                while curr != -1 and steps < 600:
                    ti.atomic_add(node_flow[curr], 0.55)
                    node_vitality[curr] = 1.0
                    curr = node_parent[curr]
                    steps += 1

    for f in range(MAX_FOODS):
        if foods[f][3] > 0.5 and food_eating_flag[f] == 1:
            foods[f][2] -= 0.6
            if foods[f][2] <= 0.0:
                foods[f][3] = 0.0

    nc = node_count[None]
    if nc > 0:
        node_vitality[0] = 1.0
        node_active[0] = 1

    for i in range(1, nc):
        if node_active[i] == 1:
            node_flow[i] *= FLOW_DECAY

            if node_flow[i] < 0.16:
                node_vitality[i] -= VITALITY_DECAY_RATE
                if node_vitality[i] <= 0.0:
                    node_vitality[i] = 0.0
                    node_active[i] = 0
            else:
                node_vitality[i] = ti.min(1.0, node_vitality[i] + 0.02)


@ti.kernel
def rasterize_vascular_network(time_val: ti.f32):
    """亚像素抗锯齿光栅化：衰退枝干线宽归零并彻底溶解"""
    for i, j in trail:
        trail[i, j] *= TRAIL_DECAY

    nc = node_count[None]
    for i in range(1, nc):
        if node_vitality[i] > 0.005:
            p_idx = node_parent[i]
            if p_idx >= 0 and node_vitality[p_idx] > 0.005:
                p0 = node_pos[p_idx]
                p1 = node_pos[i]
                dist = (p1 - p0).norm()
                steps = ti.max(1, ti.cast(ti.ceil(dist * 1.5), ti.i32))

                vit = node_vitality[i]
                flow = node_flow[i]
                depth = ti.cast(node_depth[i], ti.f32)

                thick = (0.70 * vit + flow * 2.8) * ti.exp(-0.010 * depth)
                shuttle = 1.0 + 0.20 * ti.sin(time_val * 4.0 - depth * 0.22)
                deposit = thick * vit * shuttle * 0.45

                for s in range(steps + 1):
                    t = ti.cast(s, ti.f32) / ti.cast(steps, ti.f32)
                    p = p0 * (1.0 - t) + p1 * t
                    ix = ti.cast(ti.floor(p.x), ti.i32)
                    iy = ti.cast(ti.floor(p.y), ti.i32)
                    if 1 <= ix < RES - 1 and 1 <= iy < RES - 1:
                        ti.atomic_add(trail[ix, iy], deposit)


@ti.kernel
def filter_normal_field():
    inv_9 = 1.0 / 9.0
    for i, j in trail:
        if 1 <= i < RES - 1 and 1 <= j < RES - 1:
            s = 0.0
            for dx, dy in ti.static([(-1, -1), (-1, 0), (-1, 1),
                                     (0, -1),  (0, 0),  (0, 1),
                                     (1, -1),  (1, 0),  (1, 1)]):
                s += trail[i + dx, j + dy]
            trail_blur[i, j] = s * inv_9
        else:
            trail_blur[i, j] = 0.0


@ti.kernel
def render_microscope_pipeline(time_val: ti.f32):
    center = ti.Vector([ti.cast(RES, ti.f32) * 0.5, ti.cast(RES, ti.f32) * 0.5])
    light_dir = ti.Vector([0.52, 0.68, 0.50]).normalized()
    view_dir = ti.Vector([0.0, 0.0, 1.0])
    half_vec = (light_dir + view_dir).normalized()

    col_agar = ti.Vector([0.038, 0.036, 0.032])
    col_dish_edge = ti.Vector([0.012, 0.012, 0.015])
    col_tentacle = ti.Vector([0.55, 0.60, 0.16])
    col_vein = ti.Vector([0.96, 0.82, 0.16])
    col_artery = ti.Vector([1.00, 0.98, 0.86])
    col_oat = ti.Vector([0.88, 0.82, 0.65])

    for i, j in render_buffer:
        dist_dish = (ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)]) - center).norm()

        if dist_dish > DISH_RADIUS:
            fade = ti.min(1.0, (dist_dish - DISH_RADIUS) * 0.1)
            render_buffer[i, j] = col_dish_edge * (1.0 - fade)
        else:
            im1 = ti.max(0, i - 1)
            ip1 = ti.min(RES - 1, i + 1)
            jm1 = ti.max(0, j - 1)
            jp1 = ti.min(RES - 1, j + 1)

            dx = (trail_blur[ip1, jm1] + 2.0 * trail_blur[ip1, j] + trail_blur[ip1, jp1]) - \
                 (trail_blur[im1, jm1] + 2.0 * trail_blur[im1, j] + trail_blur[im1, jp1])
            dy = (trail_blur[im1, jp1] + 2.0 * trail_blur[i, jp1] + trail_blur[ip1, jp1]) - \
                 (trail_blur[im1, jm1] + 2.0 * trail_blur[i, jm1] + trail_blur[ip1, jm1])

            normal = ti.Vector([-dx * NORMAL_BUMP, -dy * NORMAL_BUMP, 1.0]).normalized()

            half_lambert = ti.pow(normal.dot(light_dir) * 0.5 + 0.5, 2.0)
            specular = ti.pow(ti.max(0.0, normal.dot(half_vec)), 36.0)

            val = trail_blur[i, j]
            mask_tentacle = ti.min(1.0, val * 0.38)
            mask_vein = ti.pow(ti.min(1.0, val * 0.12), 2.2)
            mask_artery = ti.pow(ti.min(1.0, val * 0.035), 3.8)

            bio_col = col_agar
            bio_col = bio_col * (1.0 - mask_tentacle) + col_tentacle * mask_tentacle
            bio_col = bio_col * (1.0 - mask_vein) + col_vein * mask_vein
            bio_col = bio_col * (1.0 - mask_artery) + col_artery * mask_artery

            lit = bio_col * (0.24 + half_lambert * 0.88) + ti.Vector([1.0, 1.0, 0.95]) * (specular * 0.62 * mask_tentacle)

            # 始终绘制实体高能燕麦粒
            for f in range(MAX_FOODS):
                if foods[f][3] > 0.5:
                    df = (ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)]) - foods[f][:2]).norm()
                    r_oat = 6.5 * ti.sqrt(foods[f][2] / 25000.0)
                    if df < r_oat:
                        alpha = ti.min(0.92, foods[f][2] / 2000.0)
                        lit = lit * (1.0 - alpha) + col_oat * alpha

            # 未接种母核时，在培养皿中央显示呼吸式接种提示环
            if is_inoculated[None] == 0:
                pulse_r = 18.0 + 3.0 * ti.sin(time_val * 5.0)
                diff_r = ti.abs(dist_dish - pulse_r)
                if diff_r < 1.8:
                    alpha_pulse = (1.0 - diff_r / 1.8) * 0.75
                    lit = lit * (1.0 - alpha_pulse) + ti.Vector([0.95, 0.82, 0.20]) * alpha_pulse

            wall_margin = DISH_RADIUS - dist_dish
            if wall_margin < 10.0:
                shadow = ti.max(0.0, wall_margin / 10.0)
                lit *= (0.4 + shadow * 0.6)

            render_buffer[i, j] = ti.max(0.0, ti.min(1.0, lit))


# =========================================================================
# 系统启动主循环
# =========================================================================
def main():
    reset_simulation()

    window = ti.ui.Window(
        "Tokyo Railway Physarum Simulator v3.1",
        (RES, RES),
        vsync=True
    )
    canvas = window.get_canvas()
    gui = window.get_gui()
    dish_norm_r = DISH_RADIUS / float(RES)
    start_t = time.time()
    is_paused = False

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.LMB:
                cur = window.get_cursor_pos()
                dx, dy = cur[0] - 0.5, cur[1] - 0.5
                if dx * dx + dy * dy <= dish_norm_r * dish_norm_r:
                    px, py = cur[0] * float(RES), cur[1] * float(RES)
                    add_food_site(px, py, 30000.0)

            elif event.key == ti.ui.RMB:
                cur = window.get_cursor_pos()
                dx, dy = cur[0] - 0.5, cur[1] - 0.5
                if dx * dx + dy * dy <= dish_norm_r * dish_norm_r:
                    px, py = cur[0] * float(RES), cur[1] * float(RES)
                    inoculate_slime_mold(px, py)

            elif event.key == ti.ui.SPACE:
                is_paused = not is_paused

            elif event.key in ['r', 'R']:
                reset_simulation()

        cur_t = float(time.time() - start_t)

        # 仅在已接种且未暂停时推进生物动力学方程
        if is_inoculated[None] == 1 and not is_paused:
            bin_nodes_to_grid()
            sca_search_and_pull()
            if node_count[None] < MAX_NODES - 250:
                sca_grow_step()
            retrograde_flow_and_prune()

        rasterize_vascular_network(cur_t)
        filter_normal_field()
        render_microscope_pipeline(cur_t)

        canvas.set_image(render_buffer)

        # 屏幕常驻操作引导与状态面板
        gui.begin("Physarum Controls & Status", 0.02, 0.02, 0.40, 0.22)
        gui.text("Controls Guide:")
        gui.text("  [RMB / Right Click] : Inoculate Slime Nucleus")
        gui.text("  [LMB / Left Click]  : Place Oat Food Site")
        gui.text("  [Space]             : Pause / Resume Simulation")
        gui.text("  [R]                 : Reset Entire Dish")
        gui.text("-----------------------------------------")
        if is_inoculated[None] == 0:
            gui.text("STATUS: [Awaiting] RIGHT-CLICK inside dish to inoculate!")
        elif is_paused:
            gui.text("STATUS: [Paused]")
        else:
            gui.text(f"STATUS: [Active] Vascular Nodes: {node_count[None]} / {MAX_NODES}")
        gui.end()

        window.show()


if __name__ == "__main__":
    main()