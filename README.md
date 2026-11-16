# Real-Time XPBD Cloth Simulation Engine

A high-performance, real-time cloth and soft-body physics simulation engine built on **Extended Position-Based Dynamics (XPBD)**. Designed for stability, time-step independence, and accurate constraint solving without stiffness degradation across varying frame rates.

## Overview

Traditional Position-Based Dynamics (PBD) suffers from iteration- and time-step-dependent stiffness, often causing simulated fabrics to look overly stretchy at higher frame rates. This engine implements **XPBD**, introducing physical compliance (`α = 1/k`) and accumulated Lagrange multipliers (`λ`) to decouple material stiffness from solver iterations and `Δt`.

## Key Features

- **Time-Step Independent Stiffness:** Consistent material behavior regardless of simulation sub-steps or frame rate.
- **Constraint Solver:** Iterative Gauss-Seidel constraint projection supporting distance (stretch), bending, and attachment (pin) constraints.
- **Collision Detection & Response:** Spatial hashing for broad-phase self-collision and primitive collision handling (spheres, planes, meshes).
- **Sub-Stepping Architecture:** High-frequency sub-stepping loop for stable resolution of stiff fabrics and fast-moving objects.
- **Interactive Visualization:** Real-time rendering with camera controls, wind forces, and interactive vertex dragging.

## Mathematical Formulation

At each simulation step `Δt`, positions are predicted via symplectic Euler integration:

```text
x̃ᵢ = xᵢ + Δt · vᵢ + Δt² · wᵢ · f_ext
```

For each constraint `C` with inverse compliance `α`, the Lagrange multiplier update `Δλ` is computed as:

```text
Δλ = ( -C(x̃) - α · λ ) / ( Σ wᵢ · |∇Cᵢ|² + α / Δt² )
```

Positions and multipliers are then updated iteratively:

```text
Δxᵢ = -wᵢ · ∇Cᵢ · Δλ
λ   = λ + Δλ
```

## Getting Started

### Prerequisites
- Python 3.10+ (NumPy, Pygame / OpenGL)

### Installation & Running
```bash
git clone [https://github.com/TPOJL/Real-Time-XPBD-Cloth-Simulation-Engine.git](https://github.com/TPOJL/Real-Time-XPBD-Cloth-Simulation-Engine.git)
cd Real-Time-XPBD-Cloth-Simulation-Engine

# Install dependencies
pip install -r requirements.txt

# Run the simulation
python main.py
```

## Controls

| Input | Action |
| :--- | :--- |
| **Left Click + Drag** | Grab and move cloth vertices |
| **Right Click + Drag** | Rotate camera |
| **Space** | Pause / Resume simulation |
| **W** | Toggle wind force |
| **R** | Reset cloth state |

## References

- Macklin, M., Müller, M., & Chentanez, N. (2016). *XPBD: Position-Based Simulation of Compliant Constrained Dynamics*. Proceedings of Motion in Games (MIG).
- Müller, M., Heidelberger, B., Hennix, M., & Ratcliff, J. (2007). *Position Based Dynamics*. Journal of Visual Communication and Image Representation.

## License

This project is licensed under the MIT License.
