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
