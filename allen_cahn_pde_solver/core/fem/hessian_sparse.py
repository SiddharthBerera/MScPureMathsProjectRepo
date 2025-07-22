import torch
from torch.func import vmap, hessian
import allen_cahn_pde_solver.core.energy.allen_cahn as AllenCahnEnergy


def hessian_sparse(u, vertices, triangles, energy_fn, params):
    """
    Assemble the global sparse Hessian for a mesh using a per-triangle energy function.

    Parameters:
        u         : (N,) tensor of nodal values
        vertices  : (N, 2) tensor of node coordinates
        triangles : (T, 3) tensor of triangle indices
        energy_fn : function(u_tri, vertices_tri, *args, **kwargs) -> scalar energy
        *args, **kwargs: extra arguments for energy_fn (e.g., eps)

    Returns:
        H_sparse  : (N, N) torch.sparse_coo_tensor, global Hessian
    """
    faces_coords = vertices[triangles]   # (T, 3, 2)
    u_tri = u[triangles]                 # (T, 3)

    # Vectorized local Hessians: (T, 3, 3)
    local_hessians = vmap(hessian(energy_fn), in_dims=(0, 0, None))(
        u_tri, faces_coords, params
    )

    T, n, _ = local_hessians.shape  # n=3 for triangles

    # Assemble global indices for COO format
    rows = triangles.repeat_interleave(n, dim=1).flatten()  # (T*3*3,)
    cols = triangles.repeat(1, n).flatten()                 # (T*3*3,)
    vals = local_hessians.flatten()                         # (T*3*3,)

    indices = torch.stack([rows, cols], dim=0)
    N = u.shape[0]
    H_sparse = torch.sparse_coo_tensor(indices, vals, size=(N, N)).coalesce()
    return H_sparse