# Pure Mathematics MSc Project Repo

## 📖 Overview
This repository contains the code and computations accompanying my MSc dissertation  
**“Min–Max Methods for Computing Planar Widths and Connections to the Allen–Cahn PDE”**  
at Imperial College London, supervised by **Prof. Marco A. M. Guaraco**.

The project develops a unified **min–max** framework connecting:
- **Linear spectral theory**: min--max construction of eigenvalues  
- **Geometric widths**: nonlinear analogues of eigenvalues for length/volume functionals  
- **Allen–Cahn PDE**: variational formulation and min–max solutions

- Linear spectral theory (min--max construction of eigenvalues)  
- Geometric widths (nonlinear analogues of eigenvalues for length/volume functionals)  
- The Allen--Cahn variational PDE and its min--max solutions

## Contents
This repo includes numerical implementations (via Finite Element Methods, FEM) for two problems:

1. **Minimal Surface Evolution via the Area Functional**  
   _Folder:_ `minimal_surface_evolver`
   _Goal:_ evolve surfaces toward critical points of the area functional using gradient descent.

2. **Planar Width Exploration via the Allen–Cahn Energy**  
   _Folder:_ `allen_cahn_pde_solver`  
   _Goal:_ solve the Allen-Cahn Equation, a variational PDE, by finding min--max critical points of its associated energy via the mountain--pass algorithm.

## Quick Start
Clone:
```bash
git clone https://github.com/SiddharthBerera/MScPureMathsProjectRepo.git
cd MScPureMathsProjectRepo

---

## 📂 Repository Structure

