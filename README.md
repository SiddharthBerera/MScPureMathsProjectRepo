# Pure Mathematics MSc Project Repo

## 📖 Overview
This repository contains the code and computations accompanying my MSc dissertation *“Min–Max Methods for Computing Planar
Widths and Connections to the Allen–Cahn PDE”* at Imperial College London, supervised by Professor Marco A. M. Guaraco.  
The project develops a unified **min–max framework** connecting:

- Linear spectral theory (min–max construction of eigenvalues)  
- Geometric widths (nonlinear analogues of eigenvalues for length/volume functionals)  
- Nonlinear PDEs (Allen–Cahn equation and variational solutions)  

Programs for numerical implementation via Finite Element Methods (FEM) and visualising the results are included for two problems:
1) Minimal Surface Evolution via the Area Functional (pythonSurfaceEvolver)
2) Planar Width Exploration via the Allen--Cahn Energy (allen_cahn_pde_solver)

---

## 📂 Repository Structure
- **`/src/`** — core Python/C++ code for surface evolver–style computations  
- **`/figures/`** — generated figures used in the dissertation  
- **`/notebooks/`** — Jupyter notebooks with experiments, PDE solvers, and visualisations  
- **`/thesis/`** — LaTeX source of the dissertation  

---

## ⚙️ Installation
Clone the repository:
```bash
git clone https://github.com/SiddharthBerera/SurfaceEvolverDiss.git
cd SurfaceEvolverDiss
