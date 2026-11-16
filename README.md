# Real-Time XPBD Cloth Simulation Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Physics: XPBD](https://img.shields.io/badge/Physics-XPBD-FF5722)](https://matthias-research.github.io/pages/publications/XPBD.pdf)
[![Acceleration: NumPy & Numba](https://img.shields.io/badge/Accelerated-NumPy%20%7C%20Numba%20JIT-00A3E0)](#performance--benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

A high-performance, real-time 3D cloth physics simulation engine written in Python, built on **Extended Position-Based Dynamics (XPBD)**. This project is based on the foundational paper by **Macklin, Müller, & Chentanez (MIG 2016)** and **Matthias Müller's Ten Minute Physics #14 (*Cloth Simulation*)**.

---

## Why XPBD Instead of Classical PBD?

Traditional **Position-Based Dynamics (PBD, Müller et al., 2007)** scales constraint corrections with an ad-hoc stiffness multiplier $k \in [0, 1]$. As a consequence, effective material stiffness in PBD depends non-linearly on both the **time step $\Delta t$** and the **number of solver iterations / substeps**: increasing the iteration count or reducing $\Delta t$ makes cloth artificially rigid and heavily damped.

**Extended Position-Based Dynamics (XPBD)** solves this by introducing physical **compliance** $\alpha = 1 / k$ derived directly from an elastic energy potential $U(\mathbf{x}) = \frac{1}{2} \mathbf{C}(\mathbf{x})^T \boldsymbol{\alpha}^{-1} \mathbf{C}(\mathbf{x})$, alongside a time-step scaled compliance parameter:

$$\tilde{\alpha} = \frac{\alpha}{\Delta t^2}$$

This decouples physical material stiffness from the time step and solver iteration count, allowing completely inextensible stretching ($\alpha_{\text{stretch}} = 0$) to coexist stably with soft, tunable bending ($\alpha_{\text{bend}} \in [0, 10]$).

---

## Core Algorithmic Features (Ten Minute Physics #14)

### 1. Fast Half-Edge Mesh Topology (`findTriNeighbors`)
Rather than restricting cloth to a regular quad grid, the engine accepts arbitrary triangle meshes (default reference mesh: **3,321 vertices**, **6,400 triangles**).
- Each triangle $t$ defines 3 directed half-edges $3t + j$ ($j \in \{0, 1, 2\}$).
- Half-edges are canonically sorted in $O(M \log M)$ time to pair adjacent triangles across shared internal edges.
- **Stretching Constraints**: 1 distance constraint per unique mesh edge (**9,720 edges**).
- **Bending Constraints**: 1 constraint per adjacent triangle pair across shared internal edges (**18,960 pairs**).

### 2. Fast Isometric Bending via Opposite Vertices
Computing dihedral angle gradients $\nabla \arccos(\mathbf{n}_1 \cdot \mathbf{n}_2)$ between adjacent triangle normals is computationally expensive and prone to singularities. Following Matthias Müller's *Ten Minute Physics #14*, bending between two adjacent triangles $(i_0, i_1, i_2)$ and $(i_1, i_0, i_3)$ sharing edge $(i_0, i_1)$ is formulated as a **compliant 1D distance constraint between the two opposite vertices $(i_2, i_3)$**:

$$C_{\text{bend}}(\mathbf{x}_{i_2}, \mathbf{x}_{i_3}) = \|\mathbf{x}_{i_2} - \mathbf{x}_{i_3}\| - l_{0,\text{bend}}, \quad \alpha_{\text{bend}} > 0$$

### 3. Substepping Gauss-Seidel Solver (`numSubsteps = 15`)
Instead of taking one large frame step $\Delta t = \frac{1}{60}\text{ s}$ with 15 solver iterations, each frame is subdivided into $N_{\text{sub}} = 15$ substeps of duration $\Delta t_s = \frac{\Delta t}{N_{\text{sub}}} = \frac{1}{900}\text{ s}$ with 1 Gauss-Seidel pass per substep:

```text
For each substep s = 1 .. numSubsteps (Δt_s = Δt / numSubsteps):
  1. preSolve(Δt_s, g):
       vᵢ ← vᵢ + Δt_s · g
       x_prev,ᵢ ← xᵢ
       xᵢ ← xᵢ + Δt_s · vᵢ
       Clamp ground collision (yᵢ < 0 ⇒ xᵢ ← x_prev,ᵢ, yᵢ ← 0)

  2. solveStretching(α_stretch = 0, Δt_s) & solveBending(α_bend, Δt_s):
       α̃ = α / Δt_s²
       C = ||x₀ - x₁|| - l₀,   ĝ = (x₀ - x₁) / ||x₀ - x₁||
       Δλ = -C / (w₀ + w₁ + α̃)
       x₀ ← x₀ + w₀ · Δλ · ĝ
       x₁ ← x₁ - w₁ · Δλ · ĝ

  3. postSolve(Δt_s):
       vᵢ ← (xᵢ - x_prev,ᵢ) / Δt_s
```

### 4. Area-Weighted Masses & Interactive Raycast Grabber
- Inverse vertex masses $w_i = 1 / m_i$ are computed from adjacent triangle areas $A = \frac{1}{2} \|\mathbf{e}_0 \times \mathbf{e}_1\|$.
- Static/pinned anchors (top corners of the cloth) and interactively grabbed vertices are enforced cleanly by setting $w_i = 0$.

---

## Performance & Benchmarks

On the full reference cloth mesh from Matthias Müller's demo (**3,321 vertices**, **6,400 triangles**, **28,680 total constraints per substep** $\times$ **15 substeps** = **430,200 constraint projections per frame**):

| Backend | Frame Time (15 substeps) | Effective FPS |
| :--- | :--- | :--- |
| **NumPy + Numba `@njit` (CPU)** | **~2.4 ms / frame** | **> 400 FPS** |
| Target Real-Time Budget (60 Hz) | 16.67 ms / frame | 60 FPS |

---

## Getting Started

### Prerequisites
- Python 3.10+
- `numpy`, `numba`, `pygame`, `PyOpenGL`, `matplotlib`

### Installation & Running

```bash
git clone https://github.com/TPOJL/Real-Time-XPBD-Cloth-Simulation-Engine.git
cd Real-Time-XPBD-Cloth-Simulation-Engine

# Install dependencies
pip install -r requirements.txt

# Run the interactive 3D cloth simulation
python main.py
```

---

## Interactive Controls

| Input | Action |
| :--- | :--- |
| **Left Click + Drag** | Raycast grab and pull cloth vertices in 3D space |
| **Right Click + Drag** | Orbit 3D camera around the scene |
| **Mouse Wheel** | Zoom camera in / out |
| **Space** | Run / Pause physics simulation |
| **E** | Toggle wireframe edges vs shaded triangle mesh (`Show edges`) |
| **Up / Down Arrows** (or Slider) | Adjust `bendingCompliance` ($\alpha_{\text{bend}} \in [0, 10]$) |
| **W** | Toggle external wind force |
| **R** | Restart / reset cloth to initial state |

---

## References

1. **Macklin, M., Müller, M., & Chentanez, N. (2016).** *XPBD: Position-Based Simulation of Compliant Constrained Dynamics*. Proceedings of the 9th International Conference on Motion in Games (MIG '16), ACM, 49–54.
2. **Müller, M. (2022).** *Ten Minute Physics #14: How to Write a Cloth Simulator* (`14-cloth.html`).
3. **Müller, M., Heidelberger, B., Hennix, M., & Ratcliff, J. (2007).** *Position Based Dynamics*. Journal of Visual Communication and Image Representation, 18(2), 109–118.

---

## License

This project is licensed under the MIT License.
