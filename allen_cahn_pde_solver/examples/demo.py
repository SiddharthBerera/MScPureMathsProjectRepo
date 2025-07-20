import torch

from allen_cahn_pde_solver.core.geometry.planar_domains import TriangleDomain
from allen_cahn_pde_solver.examples.square_mesh import square_mesh
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
    energy_grad_u = E.gradient(energy, u)
    print(energy_grad_u)

    hessian_naive = hessian_dense(energy_grad_u, u, mesh.edge_list)
    print('niave O(n2) with torch')
    print(hessian_naive)
    hessian_efficient = hessian_sparse(energy_grad_u, u, mesh.edge_list)
    print('efficient with sparse hessian')
    print(hessian_efficient)


if __name__ == '__main__':
    #points   = torch.tensor([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]], dtype=torch.float64)
    #edges = torch.tensor([[0,1],[0,2],[1,2],[1,3],[2,3]])
    #faces = torch.tensor([[0,1,2],[1,2,3]])
    #border = torch.tensor([0,1,2,3])
    #u = torch.tensor([1,0,0,1], dtype=torch.float64, requires_grad=True)

    sqr_length = 1
    N = 2

    points, edges, faces, border = square_mesh(sqr_length, N)

    mesh = TriangleDomain(points, edges, faces, border)

    u = torch.rand(((N+1)**2,))

    eps = 0.1
    k=1

    demo(mesh, u, eps, k)