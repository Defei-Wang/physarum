# Physarum: Morphogenesis & Transport Network Engine

[English](https://www.google.com/search?q=%23english) | [中文说明](https://www.google.com/search?q=%23%E4%B8%AD%E6%96%87%E8%AF%B4%E6%98%8E)

A high-performance research pipeline simulating the morphogenesis, foraging dynamics, and cytoplasmic transport networks of *Physarum polycephalum* (slime mold). Built on top of Python and GPU backends (Vulkan / CUDA) via the Taichi compute infrastructure.

This repository chronicles the iterative transition from classical agent-based continuum heuristics to discrete topological space-colonization graphs, GPGPU dual-channel blending pipelines, and multi-species antagonistic 2.5D normal-mapped shading.

---

### Architecture Overview

Simulating *Physarum* morphology requires resolving a fundamental contradiction in generative systems: **local fluid continuity versus macro-scale branch pruning**. Simple particle swarms collapse into homogeneous sponge-like patterns, whereas rigid graph networks lose the fluid, organic plasticity of living protoplasm.

This codebase tracks five sequential architectural paradigms across Git tags `v1.0` ~ `v5.0`, systematically resolving transport efficiency, membrane tension, and microscopic surface relief.

```
                    ┌─────────────────────────┐
                    │ v1.0: Dense Swarm (PDE) │
                    └────────────┬────────────┘
                                 │ (Fail: concentric rings, sponge collapse)
                    ┌────────────┴────────────┐
                    ▼                         ▼
      ┌──────────────────────────┐   ┌───────────────────────────┐
      │ v2.0: Generational Buds  │   │ v3.0: SCA Topology Graph  │
      │ (Explicit tip branching) │   │ (Pure discrete attractor) │
      └────────────┬─────────────┘   └─────────────┬─────────────┘
                   │                               │
                   └─────────────┬─────────────────┘
                                 │ (Synthesis: continuum fluid + structural membrane)
                    ┌────────────▼────────────┐
                    │ v4.0: Dual-Channel Ping │
                    │ (R/G field decoupling)  │
                    └────────────┬────────────┘
                                 │ (Breakthrough: multi-species tension + 2.5D light)
                    ┌────────────▼────────────┐
                    │ v5.0: Antagonistic 2.5D │
                    │ (Sobel normal shading)  │
                    └─────────────────────────┘

```

---

### Evolutionary Lineage & Milestone Specifications

#### `v1.0` - Baseline Dense Swarm & Shuttle Streaming

* **Commit**: `2af9244`
* **Core Paradigm**: 1.2M particle continuum interacting with scalar pheromone fields.
* **Mathematical Stencil**:
* Agent heading update:

$$\theta_{t+1} = \theta_t + \Delta\theta \cdot \text{sign}(S_R - S_L)$$



where $S_L, S_C, S_R$ sample the combined trail and scent fields via three forward-offset sensors ($SO = 22.5^\circ, SA = 45.0^\circ, SS = 3.5\text{ px}$).
* Macro-scale protoplasmic shuttle streaming: Sinusoidal velocity modulation:

$$v(t) = v_0 \cdot \left[1.0 + 0.3 \sin\left(\omega t - \vec{k} \cdot \vec{x}\right)\right]$$


* Trail dissipation: $3 \times 3$ sub-pixel Gaussian kernel with multiplicative decay ($\gamma = 0.94$).


* **Pathology**: Under prolonged runs, agents lose directional divergence, clustering into static sponge-like meshes and concentric wavefront ripples upon dish boundary collisions.

#### `v2.0` - Generational Tip-Growth & Archival Paper Shader

* **Commit**: `d48f330`
* **Core Paradigm**: Explicit recursive branching with finite metabolic lifespan (inspired by tip-growth morphogenesis).
* **Mechanics**:
* Agent metadata includes generation depth $g \in [0, 6]$ and remaining lifespan $L_i$.
* Branching probability decays exponentially:

$$P_{\text{branch}} = P_0 \cdot \alpha_{\text{acc}}^g$$


* Daughter agents inherit attenuated deposition potency:

$$D_{\text{child}} = D_{\text{parent}} \cdot V_{\text{tap}}$$




* **Renderer**: Classical botanical archival paper aesthetic. Uses Fractal Brownian Motion (FBM) and pseudo-random film grain to break planar digital uniformity:

$$\text{Color} = \text{lerp}\left(\vec{C}_{\text{paper}}, \vec{C}_{\text{ink}}, \text{Trail}^\gamma\right) + \text{Grain}(\vec{x}, t)$$


* **Pathology**: Branch explosions require rigid atomic allocation counters (`alloc_counter`), sacrificing GPGPU throughput.

#### `v3.0` - Pure Topological Vascular Graph (SCA)

* **Commit**: `a8923be`
* **Core Paradigm**: Complete departure from grid-based particle fields; deployment of the Space Colonization Algorithm (SCA).
* **Mechanics**:
* Distributes 12,000 discrete nutrient attractors $\vec{A}_k$ across the dish.
* Node search: Each active attractor pulls the closest vascular node $\vec{N}_j$ within influence radius $D_{\text{attr}}$.
* Normalized growth vector accumulation:

$$\vec{v}_{\text{grow}} = \text{normalize}\left(\sum_{k} \frac{\vec{A}_k - \vec{N}_j}{\Vert{}\vec{A}_k - \vec{N}_j\Vert{}}\right)$$


* Kill distance: Attractors are consumed when $\Vert{}\vec{A}_k - \vec{N}_j\Vert{} < D_{\text{kill}}$.


* **Significance**: Mathematically prevents sponge artifacts. Enforces strict vascular hierarchy: primary trunks scale up in radius while distal search capillaries remain ultra-fine.

#### `v4.0` - Dual-Channel Delayed Blending & Torus Topology

* **Commit**: `e4941c2`
* **Core Paradigm**: 1:1 translation of classic WebGL GPGPU ping-pong framebuffer pipelines.
* **Mechanics**:
* **R Channel (Instantaneous Deposit)**: Binary rasterization of agent coordinates.
* **G Channel (Morphogenetic Memory)**: Temporal diffusion buffer.
* Convolutional mixing filter:

$$G_{t+1}(\vec{x}) = \gamma \sum_{\vec{\delta}} K(\vec{\delta}) \left[R_t(\vec{x} + \vec{\delta}) + 0.5 \cdot G_t(\vec{x} + \vec{\delta})\right]$$



with box blur weight $K = \frac{1}{9}$.
* Torus boundary wrapping via floating-point fract operations:

$$\vec{x} = \text{fract}(\vec{x})$$




* **Significance**: Zero border reflection clustering. The $1.0 : 0.5$ kernel weight creates sharp front-wave headers trailing into smooth, persistent conduits.

#### `v5.0` - Multi-Species Antagonism & 2.5D Height-Field Normal Mapping

* **Commit**: `86e1294`
* **Core Paradigm**: Multi-species interaction tensor coupled with photometric normal-bump reconstruction.
* **Mechanics**:
* **Cross-Species Tensor ($2 \times 2$)**:

$$M = \begin{pmatrix} +1.0 & -0.4 \\ -0.4 & +1.0 \end{pmatrix}$$



Intra-species positive feedback consolidates cords; inter-species negative repulsion violently tears open membrane pores, generating biological lacunae.
* **Multi-Pass Blur**: Cascaded passes ($\text{Passes} = 2$) extend spatial tension fields without sacrificing fine edges.
* **2.5D Photometric Shading**:
Treats cumulative matter $H(\vec{x}) = \sum_c \text{grid}_c(\vec{x})$ as a microscopic relief height-field:

$$\nabla H = \left(\frac{H_{x+1, y} - H_{x-1, y}}{2}, \frac{H_{x, y+1} - H_{x, y-1}}{2}\right)$$


$$\vec{N} = \text{normalize}\left(-k \cdot \partial_x H, -k \cdot \partial_y H, 1.0\right)$$


$$\text{Shading} = I_{\text{ambient}} + I_{\text{diffuse}} \max\left(0, \vec{N} \cdot \vec{L}\right)$$




* **Significance**: Achieves electron-microscope visual depth, completely replacing glowing flat particle trails with three-dimensional vascular tissue.

---

### Comparative Feature Matrix

| Version Tag | State Space | Computational Complexity | Boundary Physics | Structural Phenomenon | Visual Paradigm |
| --- | --- | --- | --- | --- | --- |
| **v1.0** | Continuum Fields | $\mathcal{O}(N_{\text{agents}})$ | Rigid Circular Reflect | Homogeneous Sponge Mesh | Dual-scale Optical Absorption |
| **v2.0** | Dynamic Agent Pool | $\mathcal{O}(N_{\text{active}})$ | Box Hard Clamp | Branching Fractal Hierarchy | Archival Ink & Paper Grain |
| **v3.0** | Node-Edge Graph | $\mathcal{O}(N_{\text{nodes}} \cdot N_{\text{attrs}})$ | Discrete Seed Region | Strict Trunk-Capillary Tree | Softened Filament Density |
| **v4.0** | Split RG Buffers | $\mathcal{O}(N_{\text{agents}} + \text{Res}^2)$ | Toroidal Periodic Wrap | High-speed Fluid Conduits | Grayscale Trail Density |
| **v5.0** | Multi-Vector Field | $\mathcal{O}(S \cdot N_{\text{agents}} + P \cdot \text{Res}^2)$ | Toroidal Periodic Wrap | Antagonistic Membrane Pores | 2.5D Directional Bump Shading |

---

### 架构全景与技术路径

模拟多头绒泡菌（*Physarum polycephalum*）的形态发生，核心难点在于**局域连续流动与宏观拓扑剪枝的动力学统一**。单纯的粒子游走模型极易退化为均质的海绵网格；而死板的图论算法又完全丧失了原生质体流动的有机质感。

本项目通过 `v1.0` ~ `v5.0` 五个里程碑版本，系统性地推演并解决了输运效率、形态开孔、表面张力与立体显像等核心问题：

```
                    ┌─────────────────────────┐
                    │  v1.0: 高密粒子连续相场 │
                    └────────────┬────────────┘
                                 │ (缺陷: 碰撞同心圆伪影、均质海绵化)
                    ┌────────────┴────────────┐
                    ▼                         ▼
      ┌──────────────────────────┐   ┌───────────────────────────┐
      │  v2.0: 显式代际顶端分叉  │   │  v3.0: 纯空间殖民拓扑图   │
      │  (末梢生长、有限寿命)   │   │  (离散养分引导、无网格)   │
      └────────────┬─────────────┘   └─────────────┬─────────────┘
                   │                               │
                   └─────────────┬─────────────────┘
                                 │ (综合路线: 流体连续性 + 拓扑自组织)
                    ┌────────────▼────────────┐
                    │  v4.0: 双通道延迟平滑   │
                    │  (R/G 解耦、环形无界)   │
                    └────────────┬────────────┘
                                 │ (终极突破: 多物种拮抗张力 + 2.5D 光照)
                    ┌────────────▼────────────┐
                    │  v5.0: 拮抗矩阵与法线场 │
                    │  (Sobel 差分立体显微质感)│
                    └─────────────────────────┘

```

---

### 版本演进编年史

#### `v1.0` - 基础高密粒子流与穿梭流 (Baseline Dense Swarm)

* **对应 Commit**: `2af9244`
* **动力学模型**：120 万粒子规模的高密度网格模型。
* **核心数学算子**：
* 三向离散嗅探偏转（$SO = 22.5^\circ, SA = 45.0^\circ, SS = 3.5\text{ px}$）。
* 原生质穿梭流行波（Shuttle Streaming）：使用低频周期正弦信号调制质点步长与分泌强度，模拟原生质宏观往复流动：

$$\text{deposit} = D_0 \cdot \left[1.0 + A \cdot \sin\left(\omega t - \vec{k} \cdot \vec{x}\right)\right]$$


* 物质场经由 $3 \times 3$ 高斯卷积平滑并施加衰减因子（$\gamma = 0.94$）。


* **失效模式分析**：系统缺乏宏观排斥机制。当粒子密度过高并碰撞培养皿边缘时，边界反射波与内部流场发生干涉，退化为同心圆环与均质海绵迷宫。

#### `v2.0` - 显式代际分叉与植物标本着色 (Generational Tip-Growth)

* **对应 Commit**: `d48f330`
* **动力学模型**：引入植物形态发生学中的顶端生长机制（Tip-Growth）。
* **核心数学算子**：
* 粒子具备代际深度 $g \in [0, 6]$ 与有限生存周期 $L_i$。
* 侧向分叉概率随代际指数级衰减：

$$P_{\text{branch}} = P_0 \cdot \alpha_{\text{acc}}^g$$


* 沿主干向上分泌强度递减，保证主干粗壮而顶芽纤细。


* **渲染着色器**：复刻古典植物档案纸张质感。利用四阶分形布朗运动（FBM）对标量场进行非均匀渗透腐蚀，叠加微观伪随机高斯噪点：

$$\text{Pixel} = \text{PaperBase} \cdot (1 - T) + \text{VascularInk} \cdot T + \text{Noise}$$


* **失效模式分析**：树状分叉机制引发粒子索引分配竞争，原子操作计数器（`atomic_add`）对并行算力造成了严重损耗。

#### `v3.0` - 纯空间殖民拓扑图网络 (Space Colonization Graph)

* **对应 Commit**: `a8923be`
* **动力学模型**：彻底摒弃连续场网格，采用经典空间殖民算法（SCA）。
* **核心数学算子**：
* 在空间中随机离散播撒 12,000 个养分吸引子（Attractors）。
* 迭代计算吸引子对局部血管树节点的欧氏距离与单位牵引向量。
* 当养分点与节点距离小于吞噬阈值 $D_{\text{kill}}$ 时，养分死亡并转化为脉管粗度。


* **形态学突破**：在数学层面上完全根绝了海绵化问题，纯粹依靠几何牵引生成具有严格粗细层级（主干到毛细）的拓扑图，保留了大量自然留白。

#### `v4.0` - 双通道延迟混合与甜甜圈环形拓扑 (Dual-Channel Blending)

* **对应 Commit**: `e4941c2`
* **动力学模型**：1:1 像素级转译 WebGL GPGPU 乒乓帧缓冲架构。
* **核心数学算子**：
* **R 通道（瞬间冲量）**：记录当前帧粒子落点的绝对离散位置。
* **G 通道（历史相场）**：平滑延时轨迹场。
* 延迟滤波混合核：将当前粒子冲量与历史相场按 $1.0 : 0.5$ 的非对称权重融合，再执行衰减：

$$G_{t+1} = \gamma \cdot \text{BoxBlur}\left(R_t + 0.5 \cdot G_t\right)$$


* 采用 `fract()` 算子构建甜甜圈环形空间（Torus Topology），坐标超出边界自动无缝穿出。


* **形态学突破**：消除了边界反弹引发的死区积压，粒子在亚像素采样下展现出如流体般的丝滑汇聚。

#### `v5.0` - 多物种拮抗矩阵与 2.5D 高度场法线渲染 (Multi-Species & 2.5D Normal)

* **对应 Commit**: `86e1294`
* **动力学模型**：多物种相互作用张量 + 光度学法线表面重构。
* **核心数学算子**：
* **交叉吸引矩阵（Attraction Tensor）**：

$$\begin{pmatrix} A_{00} & A_{01} \\ A_{10} & A_{11} \end{pmatrix} = \begin{pmatrix} +1.0 & -0.4 \\ -0.4 & +1.0 \end{pmatrix}$$



同种物质强正反馈聚拢成束，异种物质强负反馈排斥。异种切应力直接将连续相场撕开，形成天然的筋膜边缘与张力孔洞（Membrane Pores）。
* **多遍级联滤波（Multi-Pass Blur）**：执行两遍盒状平滑，在保留边缘极高频率差分的同时，将引力势场半径扩大数倍。
* **2.5D 微表面法线着色（Sobel Bump Shading）**：
将二维标量物质总量映射为虚拟三维微观地形高度 $H(x, y)$，通过有限差分提取表面法线向量：

$$\vec{N} = \text{normalize}\left( -k \frac{\partial H}{\partial x}, -k \frac{\partial H}{\partial y}, 1.0 \right)$$



引入倾斜平行光源 $\vec{L} = \text{normalize}(0.7, 0.7, 0.8)$ 计算半兰伯特漫反射与高光。


* **形态学突破**：彻底终结了扁平发光粒子的廉价数字感，画面呈现出如同冷冻电子显微镜扫描下的厚实质感与生物组织阴影。

---

### 技术演进特征横向对比

| 评估维度 | v1.0 密集粒子流 | v2.0 顶芽分叉 | v3.0 空间殖民图 | v4.0 双通道混合 | v5.0 拮抗法线场 |
| --- | --- | --- | --- | --- | --- |
| **状态载体** | 单层连续标量场 | 粒子属性结构体 | 树状图节点列表 | R/G 分离缓冲 | 双物种向量场 |
| **空间拓扑** | 培养皿刚性反射 | 自由边界硬截断 | 离散有限有界空间 | 环形无缝甜甜圈 | 环形无缝甜甜圈 |
| **形态缺陷** | 易产生海绵化斑图 | 粒子数易超载爆炸 | 计算复杂度偏高 | 缺乏立体张力空洞 | 计算负荷达全管线峰值 |
| **视觉呈现** | 扁平光学吸收伪彩 | 古典标本纸张暗调 | 平滑骨架线条 | 灰度亚像素拖尾 | 电子显微镜 2.5D 浮雕 |

---

### 环境配置与运行指南 (Setup & Quick Start)

#### 1. 运行时依赖

* 操作系统: Windows 10/11, Linux, macOS
* Python 环境: Python 3.8 ~ 3.12
* 底层硬件支持: NVIDIA GPU (推荐，调用 CUDA 后端) 或支持 Vulkan 1.2+ 的核显/独显

安装核心依赖库：

```bash
pip install taichi numpy

```

#### 2. 检出并运行历史里程碑

克隆仓库后，可通过 Git Tag 自由穿梭检视五个演进阶段的物理特性：

```bash
# 运行 v1.0 基础穿梭流模型
git checkout v1.0
python main.py

# 运行 v3.0 空间殖民拓扑血管网
git checkout v3.0
python main.py

# 运行 v4.0 双通道丝滑流体
git checkout v4.0
python main.py

# 运行 v5.0 终极多物种 2.5D 法线立体微观组织
git checkout v5.0
python main.py

```

#### 3. 交互操作定义

* **鼠标左键 (LMB)**: 向培养皿当前点位注入养分/高浓度粒子团，诱发网络形态重组。
* **按键 R**: 清空当前相场显存，重新进行原核接种初始化。

---

### 核心参考文献与理论源流 (References)

* Jeff Jones. *Characteristics of Pattern Formation and Evolution in Approximations of Physarum Polycephalum*. Artificial Life, 2010.
* Adam Runions, Martin Fuhrer, Brendan Lane, Pavol Federl, Anne-Gaelle Rolland-Lagan, Przemyslaw Prusinkiewicz. *Modeling and visualization of leaf venation patterns*. ACM Transactions on Graphics (TOG), 2005. (Space Colonization Algorithm)
* Sage Jenson. *Physarum Polycephalum Simulation & Procedural Aesthetics*. `[https://cargocollective.com/sagejenson/physarum](https://cargocollective.com/sagejenson/physarum)`
* Nicolas Barradeau (nicoptere). *WebGL GPGPU Multi-agent Transport Network*. `[https://github.com/nicoptere/physarum](https://github.com/nicoptere/physarum)`
* Michael Fogleman. *Physarum - Multi-species Agent Simulation in Go*. `[https://github.com/fogleman/physarum](https://github.com/fogleman/physarum)`
