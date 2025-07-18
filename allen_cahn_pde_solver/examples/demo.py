import torch
from allen_cahn_pde_solver.core.geometry.planar_domains import TriangleDomain
from allen_cahn_pde_solver.core.fem.torch_assembly import assemble_basis
from allen_cahn_pde_solver.core.energy.allen_cahn import AllenCahnEnergy

def demo(mesh: TriangleDomain, u: torch.tensor, eps: float, k: int):
    
    triangles= torch.tensor([[0,1,2], [1,2,3]], dtype=torch.long)

    u = u.clone().detach().requires_grad_(True)


    # 2) Assemble Pk Basis for approximating u over triangles of the mesh
    areas, basis_matrices = assemble_basis(points, triangles, k)
    #print(basis_matrices)
    #print(areas)

    # 3) Allen–Cahn energy
    E   = AllenCahnEnergy(areas, basis_matrices, eps)

    # 5) Compute and print
    energy = E.value(u, triangles)
    print(energy)
    energy_grad_u = E.gradient(energy, u)
    print(energy_grad_u)
    hessian = E.hessian_efficient(energy_grad_u, u, mesh.edge_list)
    print(hessian)

if __name__ == '__main__':
    points   = torch.tensor([[0.0, 0.0],[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]], dtype=torch.float64)
    edges = torch.tensor([[0,1],[0,2],[1,2],[1,3],[2,3]])
    faces = torch.tensor([[0,1,2],[1,2,3]])
    borders = torch.tensor([0,1,2,3])
    u = torch.tensor([1,0,0,1], dtype=torch.float64, requires_grad=True)

    triangle = TriangleDomain(points, edges, faces, borders)

    eps = 0.1
    k=1

    demo(triangle, u, eps, k)
