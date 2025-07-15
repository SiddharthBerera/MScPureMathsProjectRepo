import torch
from allen_cahn_pde_solver.core.geometry.planar_domains import TriangleDomain
from allen_cahn_pde_solver.core.fem.torch_assembly import assemble_basis
from allen_cahn_pde_solver.core.energy.allen_cahn import AllenCahnEnergy

def demo(u: torch.tensor, eps: float, k: int):
    points   = torch.tensor([[0.0, 0.0],
                             [1.0, 0.0],
                             [0.0, 1.0],
                             [1.0, 1.0]], dtype=torch.float64)
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
    energy.backward()
    grad_norm = u.grad.norm()
    print(grad_norm)

if __name__ == '__main__':
    u = torch.tensor([1,0,0,1], dtype=torch.float64, requires_grad=True)
    eps = 0.1
    k=1

    demo(u, eps, k)
