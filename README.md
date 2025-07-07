# SurfaceEvolverDiss
_A command-line toolkit for exploring Allen–Cahn min–max solutions and
$p$-widths on triangulable domains.

## 1.  Project scope

This project re-implements Brakke’s *Surface Evolver* philosophy for
phase-transition geometry:

* **Allen–Cahn energy** on 2-D domains tessellated by equilateral
  triangles  
  $E_\varepsilon(u)=\int_{\Omega}(\tfrac{\varepsilon}{2}|\nabla u|^{2}+
  \tfrac{(1-u^{2})^{2}}{4\varepsilon})\,dx$.
* **Interactive CLI** that lets a user
  - assemble FEM matrices,
  - run gradient flow, Newton steps, or a $k$-parameter mountain-pass
    loop,
  - verify Morse index and undo/redo along an evolution history.
* **Finite-dimensional subspaces** spanned by the first \(k\) Dirichlet
  Laplacian eigenfunctions so the resulting critical point realises the
  $k$-width of the domain (Dey 2024)
