# comparison.py

import torch
import numpy as np
import time
from torch.func import vmap, hessian
from scipy.sparse import csc_matrix

from allen_cahn_pde_solver.core.fem.torch_assembly import assemble_basis
from allen_cahn_pde_solver.core.energy.allen_cahn import AllenCahnEnergy
from allen_cahn_pde_solver.core.fem.gradient import gradient
from allen_cahn_pde_solver.core.fem.hessian_sparse import hessian_sparse


class MinmaxerEnergy:
    """
    Computes the Allen-Cahn energy using an element-wise, vmap-based approach.
    """

    def __init__(self, eps: float):
        self.eps = eps

    def _dirichlet_energy_single(self, vertices, u_vals):
        v0, v1, v2 = vertices
        e1, e2 = v1 - v0, v2 - v0
        area = 0.5 * torch.abs(e1[0] * e2[1] - e1[1] * e2[0])
        M = torch.cat([vertices, torch.ones((3, 1), dtype=vertices.dtype, device=vertices.device)], dim=1)
        try:
            Minv = torch.linalg.inv(M)
        except torch.linalg.LinAlgError:
            return torch.tensor(0.0, dtype=vertices.dtype, device=vertices.device)
        abc = torch.matmul(Minv, u_vals.unsqueeze(1)).squeeze()
        a, b = abc[0], abc[1]
        return 0.5 * (a**2 + b**2) * area

    def _potential_energy_single(self, vertices, u_vals):
        u1, u2, u3 = u_vals
        v0, v1, v2 = vertices
        e1, e2 = v1 - v0, v2 - v0
        area = 0.5 * torch.abs(e1[0] * e2[1] - e1[1] * e2[0])
        poly = (
            15 + u1**4 + u2**4 + u2**3*u3 - 5*u3**2 + u3**4 + u1**3*(u2 + u3) +
            u2**2*(-5 + u3**2) + u2*u3*(-5 + u3**2) +
            u1*(u2 + u3)*(-5 + u2**2 + u3**2) +
            u1**2*(-5 + u2**2 + u2*u3 + u3**2)
        )
        return area / 15.0 * poly

    def _energy_single_triangle(self, vertices, u_vals):
        dir_integral = self._dirichlet_energy_single(vertices, u_vals)
        pot_integral = self._potential_energy_single(vertices, u_vals)
        return self.eps * dir_integral + (1.0 / (4.0 * self.eps)) * pot_integral

    def value(self, u, vertices, triangles):
        faces_coords = vertices[triangles]
        u_tri = u[triangles]
        energies_per_triangle = vmap(self._energy_single_triangle)(faces_coords, u_tri)
        return torch.sum(energies_per_triangle)

    def gradient(self, energy, u):
        return torch.autograd.grad(energy, u, create_graph=True, retain_graph=True)[0]
    
    def hessian(self, u, vertices, triangles):
        num_vertices = vertices.shape[0]
        faces_coords = vertices[triangles]
        u_tri = u[triangles]
        def energy_fn_for_hess(u_v, v_coords):
            return self._energy_single_triangle(v_coords, u_v)
        local_hessians = vmap(hessian(energy_fn_for_hess), in_dims=(0, 0))(u_tri, faces_coords)
        num_faces, num_dofs_per_face, _ = local_hessians.shape
        rows = triangles.repeat_interleave(num_dofs_per_face, dim=1).flatten()
        cols = triangles.repeat(1, num_dofs_per_face).flatten()
        vals = local_hessians.flatten()
        indices = torch.stack([rows, cols], dim=0)
        H_sparse_coo = torch.sparse_coo_tensor(indices, vals, size=(num_vertices, num_vertices)).coalesce()
        indices_np = H_sparse_coo.indices().cpu().numpy()
        values_np = H_sparse_coo.values().detach().cpu().numpy()
        return csc_matrix((values_np, (indices_np[0], indices_np[1])), shape=(num_vertices, num_vertices))

    def hessian_naive(self, u, vertices, triangles):
        def total_energy_fn(u_vec):
            return self.value(u_vec, vertices, triangles)
        return hessian(total_energy_fn)(u)

# --- Parameters ---
N = 6
NUM_TIMING_RUNS = 5

def create_subdivided_triangle_mesh(n_subdivisions, size=10.0):
    height = size * np.sqrt(3) / 2
    vertices = torch.tensor([[0.0, 0.0], [size, 0.0], [size / 2, height]], dtype=torch.float64)
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.long)
    for _ in range(n_subdivisions):
        new_triangles, midpoint_cache, vertex_list = [], {}, vertices.tolist()
        for tri_indices in triangles:
            v1_idx, v2_idx, v3_idx = tri_indices.tolist()
            edge_midpoint_indices = []
            for i, j in [(v1_idx, v2_idx), (v2_idx, v3_idx), (v3_idx, v1_idx)]:
                edge_key = tuple(sorted((i, j)))
                if edge_key not in midpoint_cache:
                    mid_point_coords = (vertices[i] + vertices[j]) / 2.0
                    vertex_list.append(mid_point_coords.tolist())
                    new_midpoint_idx = len(vertex_list) - 1
                    midpoint_cache[edge_key] = new_midpoint_idx
                edge_midpoint_indices.append(midpoint_cache[edge_key])
            m12_idx, m23_idx, m31_idx = edge_midpoint_indices
            new_triangles.extend([[v1_idx, m12_idx, m31_idx], [v2_idx, m23_idx, m12_idx],
                                  [v3_idx, m31_idx, m23_idx], [m12_idx, m23_idx, m31_idx]])
        vertices = torch.tensor(vertex_list, dtype=torch.float64)
        triangles = torch.tensor(new_triangles, dtype=torch.long)
    return vertices, triangles

def get_edges_from_triangles(triangles):
    edges = torch.cat([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], dim=0)
    return torch.unique(torch.sort(edges, dim=1)[0], dim=0)

if __name__ == '__main__':
    # --- Part 1: Verification Run ---
    print("--- Part 1: Verification of Results on a Single Function ---")
    mesh_vertices, mesh_triangles = create_subdivided_triangle_mesh(N)
    print(f"\nNumber of subdivisions (N): {N}")
    print(f"Resulting number of vertices: {mesh_vertices.shape[0]}")
    print(f"Resulting number of triangles: {mesh_triangles.shape[0]}")

    EPS = 0.1
    params = {'eps': EPS}
    num_vertices = mesh_vertices.shape[0]
    u_base = torch.randn(num_vertices, dtype=torch.float64)

    # --- AllenCahnEnergy (Vectorized Global) ---
    print("\n[Method 1] AllenCahnEnergy (Vectorized Global)")
    u_ac = u_base.clone().requires_grad_(True)
    areas, basis_matrices = assemble_basis(mesh_vertices, mesh_triangles, k=1)
    E_ac_obj = AllenCahnEnergy(areas, basis_matrices, eps=EPS)
    energy_ac = E_ac_obj.value(u_ac, mesh_triangles)
    grad_ac = gradient(energy_ac, u_ac)
    mesh_edges = get_edges_from_triangles(mesh_triangles)
    hessian_ac = hessian_sparse(u_ac, mesh_vertices, mesh_triangles, AllenCahnEnergy.triangle_energy, params)
    
    print(f"  Total energy: {energy_ac.item():.6f}")
    print(f"  Gradient norm: {torch.linalg.norm(grad_ac).item():.6f}")
    print(f"  Hessian norm: {torch.linalg.norm(hessian_ac.to_dense()).item():.6f}")

    # --- MinmaxerEnergy (Element-wise Assembly) ---
    print("\n[Method 2] MinmaxerEnergy (Element-wise Sparse Assembly)")
    u_mm = u_base.clone().requires_grad_(True)
    E_mm_obj = MinmaxerEnergy(eps=EPS)
    hessian_mm_sparse = E_mm_obj.hessian(u_mm, mesh_vertices, mesh_triangles)
    
    energy_mm = E_mm_obj.value(u_mm, mesh_vertices, mesh_triangles)
    grad_mm = E_mm_obj.gradient(energy_mm, u_mm)
    print(f"  Total energy: {energy_mm.item():.6f}")
    print(f"  Gradient norm: {torch.linalg.norm(grad_mm).item():.6f}")
    print(f"  Hessian norm: {np.linalg.norm(hessian_mm_sparse.toarray()):.6f}")

    # --- Naive Hessian (Dense Autograd) ---
    print("\n[Method 3] MinmaxerEnergy (Naive Dense Autograd) -- Ground Truth")
    u_naive = u_base.clone().requires_grad_(True)
    hessian_mm_naive = E_mm_obj.hessian_naive(u_naive, mesh_vertices, mesh_triangles)
    print(f"  Hessian norm: {torch.linalg.norm(hessian_mm_naive).item():.6f}")

    # --- Hessian Comparison (Corrected) ---
    print("\n--- Hessian Matrix Comparison (vs Naive Ground Truth) ---")
    hessian_mm_dense_from_sparse = torch.from_numpy(hessian_mm_sparse.toarray()).to(hessian_mm_naive.device, dtype=hessian_mm_naive.dtype)
    diff_naive_vs_ac = torch.linalg.norm(hessian_mm_naive - hessian_ac)
    diff_naive_vs_sparse = torch.linalg.norm(hessian_mm_naive - hessian_mm_dense_from_sparse)
    print(f"  Norm of difference (Naive dense vs. AC efficient): {diff_naive_vs_ac.item():.6e}")
    print(f"  Norm of difference (Naive dense vs. Minmaxer sparse): {diff_naive_vs_sparse.item():.6e}")

    # --- Part 2: Timing Analysis ---
    print("\n--- Part 2: Hessian Computation Timings (avg over 10 runs) ---")
    
    # --- Timing AllenCahnEnergy ---
    ac_times = []
    for _ in range(NUM_TIMING_RUNS):
        u_rand = torch.randn(num_vertices, dtype=torch.float64, requires_grad=True)
        energy_ = E_ac_obj.value(u_rand, mesh_triangles)
        grad_ = gradient(energy_, u_rand)
        start_time = time.perf_counter()
        _ = hessian_sparse(u_rand, mesh_vertices, mesh_triangles, AllenCahnEnergy.triangle_energy, params)
        end_time = time.perf_counter()
        ac_times.append(end_time - start_time)
    avg_ac_ms = (sum(ac_times) / NUM_TIMING_RUNS) * 1000
    print(f"[Method 1] Avg time for AllenCahnEnergy (efficient): {avg_ac_ms:.4f} ms")

    # --- Timing MinmaxerEnergy (Sparse) ---
    mm_sparse_times = []
    for _ in range(NUM_TIMING_RUNS):
        u_rand = torch.randn(num_vertices, dtype=torch.float64, requires_grad=True)
        start_time = time.perf_counter()
        _ = E_mm_obj.hessian(u_rand, mesh_vertices, mesh_triangles)
        end_time = time.perf_counter()
        mm_sparse_times.append(end_time - start_time)
    avg_mm_sparse_ms = (sum(mm_sparse_times) / NUM_TIMING_RUNS) * 1000
    print(f"[Method 2] Avg time for MinmaxerEnergy (sparse assembly): {avg_mm_sparse_ms:.4f} ms")

    # --- Timing MinmaxerEnergy (Naive) ---
    mm_naive_times = []
    for _ in range(NUM_TIMING_RUNS):
        u_rand = torch.randn(num_vertices, dtype=torch.float64, requires_grad=True)
        start_time = time.perf_counter()
        _ = E_mm_obj.hessian_naive(u_rand, mesh_vertices, mesh_triangles)
        end_time = time.perf_counter()
        mm_naive_times.append(end_time - start_time)
    avg_mm_naive_ms = (sum(mm_naive_times) / NUM_TIMING_RUNS) * 1000
    print(f"[Method 3] Avg time for MinmaxerEnergy (naive dense): {avg_mm_naive_ms:.4f} ms")