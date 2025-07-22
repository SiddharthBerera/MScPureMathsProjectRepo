import torch
import time

from allen_cahn_pde_solver.core.geometry.planar_domains import TriangleDomain
from examples.square_mesh import square_mesh
from allen_cahn_pde_solver.core.fem.torch_assembly import assemble_basis
from allen_cahn_pde_solver.core.fem.gradient import gradient
from allen_cahn_pde_solver.core.fem.hessian_dense import hessian_dense
from allen_cahn_pde_solver.core.fem.hessian_sparse import hessian_sparse
from allen_cahn_pde_solver.core.energy.allen_cahn import AllenCahnEnergy


def demo(mesh: TriangleDomain, u: torch.tensor, eps: float, k: int):
    
    u = u.clone().detach().requires_grad_(True)
    print(f'u {u}')

    # 2) Assemble Pk Basis for approximating u over triangles of the mesh
    areas, basis_matrices = assemble_basis(points, mesh.face_list, k)
    #print(basis_matrices)
    #print(areas)

    # 3) Allen–Cahn energy
    E = AllenCahnEnergy(areas, basis_matrices, eps)

    # 5) Compute and print
    energy = E.value(u, mesh.face_list)
    print(energy)

    energy_grad_u = gradient(energy, u)
    print(energy_grad_u)

    t0 = time.perf_counter()
    hessian_naive = hessian_dense(energy_grad_u, u, mesh.edge_list)
    t1 = time.perf_counter()
    print('niave O(n2) with torch')
    #print(hessian_naive)
    print(f"Naive dense Hessian time: {(t1-t0)*1000:.3f} ms")


    params = {'eps': eps}
    t2 = time.perf_counter()
    hessian_efficient = hessian_sparse(u, mesh.vertex_list, mesh.face_list, AllenCahnEnergy.triangle_energy, params)
    t3 = time.perf_counter()
    print('efficient with sparse hessian')
    #print(hessian_efficient)
    print(f"Efficient sparse Hessian time: {(t3-t2)*1000:.3f} ms")


    hessian_efficient_dense = hessian_efficient.to_dense()
    print(torch.allclose(hessian_naive, hessian_efficient_dense, atol=1e-8))
    print("Max abs diff:", (hessian_naive - hessian_efficient_dense).abs().max())


if __name__ == '__main__':
    #points   = torch.tensor([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]], dtype=torch.float64)
    #edges = torch.tensor([[0,1],[0,2],[1,2],[1,3],[2,3]])
    #faces = torch.tensor([[0,1,2],[1,2,3]])
    #border = torch.tensor([0,1,2,3])
    #u = torch.tensor([1,0,0,1], dtype=torch.float64, requires_grad=True)

    sqr_length = 1
    N = 60

    points, edges, faces, border = square_mesh(sqr_length, N)

    mesh = TriangleDomain(points, edges, faces, border)

    u = torch.rand(((N+1)**2,), dtype=torch.float64, requires_grad=True)

    eps = 0.1
    k=1

    demo(mesh, u, eps, k)