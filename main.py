"""
Physarum Polycephalum - Centrifugal Exploration & Centripetal Retrograde
终极物理修正：引入离心拓荒场打破中心内卷死锁，重塑致密的毛细扩张波前。
"""

import taichi as ti
import math

ti.init(arch=ti.cuda, fast_math=True)

RES = 850
N_AGENTS = 600000      # 60万全量并发
MAX_FOODS = 32
CENTER = ti.Vector([RES / 2.0, RES / 2.0])
RADIUS = RES / 2.0 - 6.0
CAP = 5.0              

trail = ti.field(dtype=ti.f32, shape=(RES, RES))
trail_new = ti.field(dtype=ti.f32, shape=(RES, RES))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))

pos = ti.Vector.field(2, dtype=ti.f32, shape=N_AGENTS)
heading = ti.field(dtype=ti.f32, shape=N_AGENTS)
state = ti.field(dtype=ti.i32, shape=N_AGENTS)        # 0:死亡, 1:离心探索者, 2:向心回流者
life = ti.field(dtype=ti.f32, shape=N_AGENTS)         
alloc_ptr = ti.field(dtype=ti.i32, shape=())          

foods = ti.Vector.field(4, dtype=ti.f32, shape=MAX_FOODS)
food_cursor = ti.field(dtype=ti.i32, shape=())

@ti.func
def rot(v, th):
    c, s = ti.cos(th), ti.sin(th)
    return ti.Vector([v.x * c - v.y * s, v.x * s + v.y * c])

@ti.func
def clamp_idx(v, n):
    return ti.min(ti.max(v, 0), n - 1)

@ti.func
def sample_field(f: ti.template(), p):
    x = ti.min(ti.max(p.x, 0.0), RES - 1.001)
    y = ti.min(ti.max(p.y, 0.0), RES - 1.001)
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = x - x0, y - y0
    return f[x0, y0] * (1 - fx) * (1 - fy) + \
           f[x1, y0] * fx * (1 - fy) + \
           f[x0, y1] * (1 - fx) * fy + \
           f[x1, y1] * fx * fy

@ti.kernel
def init_simulation():
    alloc_ptr[None] = 0
    food_cursor[None] = 0
    for i in range(N_AGENTS):
        state[i] = 0
    for i in range(MAX_FOODS):
        foods[i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
    for i, j in trail:
        trail[i, j] = 0.0
        trail_new[i, j] = 0.0

@ti.kernel
def pump_core_agents():
    # 高频泵出盲探者，提供源源不断的开荒大军
    for _ in range(2500):  
        idx = ti.atomic_add(alloc_ptr[None], 1) % N_AGENTS
        state[idx] = 1     
        life[idx] = 500.0  
        ang = ti.random(ti.f32) * 2.0 * math.pi
        r = ti.sqrt(ti.random(ti.f32)) * 15.0
        pos[idx] = ti.Vector([CENTER.x + ti.cos(ang) * r, CENTER.y + ti.sin(ang) * r])
        heading[idx] = ang + (ti.random(ti.f32) - 0.5) * 1.0

@ti.func
def get_food_scent(p):
    s = 0.0
    for f in range(MAX_FOODS):
        if foods[f][3] > 0.5:
            dist = (p - foods[f].xy).norm()
            # 严格盲探限制：55像素外彻底断绝食物信号
            if dist < 55.0:
                s += 6.0 * (1.0 - dist / 55.0)  
    return s

@ti.func
def get_outward_scent(p):
    # 【核心解药：离心拓荒力】
    # 探索者极其渴望远离中心，这股力量将强势撕裂中心内卷，把探索网像伞一样撑开！
    return (p - CENTER).norm() * 0.025

@ti.func
def get_home_scent(p):
    # 【向心回流力】
    # 回流者渴望回归中心母核，形成笔直输运管道
    return (1000.0 - (p - CENTER).norm()) * 0.05

@ti.kernel
def step_agents(so: ti.f32, sa: ti.f32, ra: ti.f32, ss: ti.f32, dep_exp: ti.f32, dep_ret: ti.f32):
    for i in range(N_AGENTS):
        st = state[i]
        if st > 0:
            life[i] -= 1.0
            if life[i] <= 0.0:
                state[i] = 0  
                continue

            p = pos[i]
            h = heading[i]
            d = ti.Vector([ti.cos(h), ti.sin(h)])

            pf = p + d * so
            pl = p + rot(d, sa) * so
            pr = p + rot(d, -sa) * so

            Fv = sample_field(trail, pf)
            Lv = sample_field(trail, pl)
            Rv = sample_field(trail, pr)

            if st == 1:
                # 盲探者：躲避同类高浓度轨迹(抗内卷) + 受离心力驱使 + 受食物短程吸引
                # 将轨迹引力打个折，防止过度塌缩成一根线
                Fv = Fv * 0.5 + get_outward_scent(pf) + get_food_scent(pf)
                Lv = Lv * 0.5 + get_outward_scent(pl) + get_food_scent(pl)
                Rv = Rv * 0.5 + get_outward_scent(pr) + get_food_scent(pr)
            else:
                # 回流者：强力顺应轨迹通道 + 强力回归母核
                Fv += get_home_scent(pf)
                Lv += get_home_scent(pl)
                Rv += get_home_scent(pr)

            turn = 0.0
            if Fv > Lv and Fv > Rv:
                turn = 0.0
            elif Fv < Lv and Fv < Rv:
                turn = ra if ti.random(ti.f32) < 0.5 else -ra
            elif Lv < Rv:
                turn = -ra  
            else:
                turn = ra   
            
            # 探索者施加巨大的布朗扰动，让网像树枝一样分叉
            jitter = 0.45 if st == 1 else 0.05
            turn += (ti.random(ti.f32) - 0.5) * jitter 
            
            h2 = h + turn
            d2 = ti.Vector([ti.cos(h2), ti.sin(h2)])
            p2 = p + d2 * ss

            # ==== 物理相变触网 ====
            if st == 1:
                for f in range(MAX_FOODS):
                    if foods[f][3] > 0.5:
                        if (p2 - foods[f].xy).norm_sqr() < 100.0:  
                            foods[f][2] -= 0.6                     
                            if foods[f][2] <= 0.0:
                                foods[f][3] = 0.0
                            
                            # 触碰到食物，瞬间掉头，变身大动脉回流！
                            state[i] = 2
                            life[i] = 1200.0  
                            h2 += 3.14159265 + (ti.random(ti.f32) - 0.5) * 0.3
                            break
            elif st == 2:
                # 卸载能量回母核
                if (p2 - CENTER).norm_sqr() < 400.0:
                    state[i] = 0
                    continue

            # 培养皿边界死亡惩罚
            rel = p2 - CENTER
            dist = rel.norm()
            if dist > RADIUS:
                n = rel.normalized()
                d2 = d2 - 2.0 * d2.dot(n) * n
                p2 = CENTER + n * (RADIUS - 1.0)
                h2 = ti.atan2(d2.y, d2.x)
                life[i] -= 35.0  

            pos[i] = p2
            heading[i] = h2

            ix, iy = int(p2.x), int(p2.y)
            if 0 <= ix < RES and 0 <= iy < RES:
                dep_amt = dep_exp if st == 1 else dep_ret
                ti.atomic_add(trail[ix, iy], dep_amt * (1.0 - ti.min(1.0, trail[ix, iy] / CAP)))

@ti.kernel
def diffuse_decay(diffuse_mix: ti.f32, decay: ti.f32):
    for i, j in trail:
        s = 0.0
        for dx, dy in ti.static([(-1, -1), (-1, 0), (-1, 1), 
                                 (0, -1),  (0, 0),  (0, 1), 
                                 (1, -1),  (1, 0),  (1, 1)]):
            ni = clamp_idx(i + dx, RES)
            nj = clamp_idx(j + dy, RES)
            s += trail[ni, nj]
        
        blur = s / 9.0
        val = trail[i, j] * (1.0 - diffuse_mix) + blur * diffuse_mix
        val *= (1.0 - decay)
        trail_new[i, j] = ti.min(val, CAP)

@ti.func
def palette(v):
    v = ti.min(ti.max(v, 0.0), 1.0)
    v = ti.pow(v, 0.62)  
    dark = ti.Vector([0.010, 0.015, 0.012])
    olive = ti.Vector([0.18, 0.35, 0.08])
    bright = ti.Vector([0.98, 0.92, 0.25])
    
    col = ti.Vector([0.0, 0.0, 0.0])
    if v < 0.40:
        col = dark * (1.0 - v / 0.40) + olive * (v / 0.40)
    else:
        col = olive * (1.0 - (v - 0.40) / 0.60) + bright * ((v - 0.40) / 0.60)
    return col

@ti.kernel
def render_2p5d_relief(bump_scale: ti.f32):
    for i, j in trail:
        p = ti.Vector([float(i), float(j)])
        dist = (p - CENTER).norm()
        
        if dist > RADIUS:
            pixels[i, j] = ti.Vector([0.012, 0.012, 0.012])
        else:
            ip, im = clamp_idx(i + 1, RES), clamp_idx(i - 1, RES)
            jp, jm = clamp_idx(j + 1, RES), clamp_idx(j - 1, RES)
            
            dHdx = (trail[ip, j] - trail[im, j]) * 0.5
            dHdy = (trail[i, jp] - trail[i, jm]) * 0.5
            
            normal = ti.Vector([-bump_scale * dHdx, -bump_scale * dHdy, 1.0]).normalized()
            light = ti.Vector([0.35, 0.55, 0.75]).normalized()
            
            ndotl = ti.max(0.0, normal.dot(light))
            base_col = palette(trail[i, j] / CAP)
            
            shade = 0.45 + 0.75 * ndotl
            col = base_col * ti.min(shade, 1.35)
            
            for f in range(MAX_FOODS):
                if foods[f][3] > 0.5:
                    df = (p - foods[f].xy).norm()
                    r_oat = 6.0 * ti.sqrt(foods[f][2] / 500.0)
                    if df < r_oat:
                        alpha = ti.min(0.95, foods[f][2] / 100.0)
                        oat_col = ti.Vector([0.90, 0.82, 0.65])
                        col = col * (1.0 - alpha) + oat_col * alpha * shade
            
            pixels[i, j] = ti.min(col, 1.0)

def main():
    init_simulation()
    window = ti.ui.Window("Centrifugal Exploration & Retrograde Flow [LMB: Oat | R: Reset]", (RES, RES), vsync=True)
    canvas = window.get_canvas()
    
    so = 13.0
    sa = math.radians(45.0)
    ra = math.radians(45.0)
    ss = 1.25
    
    # 极低探索分泌(保证毛细不内卷)，极高回流分泌(瞬间点亮主动脉)
    dep_explore = 0.05  
    dep_return = 2.5    
    
    diffuse_mix = 0.35 
    decay = 0.015 # 保持网格生长的历史遗迹
    
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                mx, my = window.get_cursor_pos()
                px, py = mx * RES, my * RES
                idx = food_cursor[None] % MAX_FOODS
                foods[idx] = ti.Vector([px, py, 3000.0, 1.0]) 
                food_cursor[None] += 1
            elif e.key == 'r':
                init_simulation()
                
        pump_core_agents()
        step_agents(so, sa, ra, ss, dep_explore, dep_return)
        diffuse_decay(diffuse_mix, decay)
        trail.copy_from(trail_new)
        render_2p5d_relief(bump_scale=2.0)
        
        canvas.set_image(pixels)
        window.show()

if __name__ == "__main__":
    main()