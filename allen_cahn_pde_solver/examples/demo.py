import torch
from allen_cahn_pde_solver.core.geometry.planar_domains import TriangleDomain
from allen_cahn_pde_solver.core.fem.torch_assembly import assemble_mass_and_stiffness
from allen_cahn_pde_solver.core.energy.allen_cahn import AllenCahnEnergy
'''
def demo():
    # 1. Create a simple triangular domain
    dom = TriangleDomain(vertices=[(0, 0), (1, 0), (0, 1)])
    # 2. Generate mesh with max edge length h
    points, triangles = dom.mesh(h=0.1)
    # 3. Assemble mass and stiffness matrices
    M, K = assemble_mass_and_stiffness(points, triangles)
    # 4. Instantiate Allen-Cahn energy
    eps = 0.1
    E = AllenCahnEnergy(M, K, eps)
    # 5. Initialize nodal values u
    u = torch.zeros(points.shape[0], requires_grad=True)
    # 6. Compute energy and gradient norm
    energy = E.value(u)
    energy.backward()
    grad_norm = u.grad.norm()
    # 7. Print results
    print(f"Computed Allen-Cahn energy at u=0: {energy.item():.6f}")
    print(f"Gradient norm at u=0: {grad_norm.item():.6f}")
'''
def demo():
    points   = torch.tensor([[0.0, 0.0],
                             [1.0, 0.0],
                             [0.0, 1.0]], dtype=torch.float64)
    triangles= torch.tensor([[0,1,2]], dtype=torch.long)
    k=1

    # 2) Assemble M, K
    areas, basis_matrices = assemble_mass_and_stiffness(points, triangles, k)

    # 3) Allen–Cahn energy
    eps = 0.1
    E   = AllenCahnEnergy(areas, basis_matrices, eps)

    # 4) Zero field
    u = torch.zeros(points.shape[0], dtype=torch.float64, requires_grad=True)

    # 5) Compute and print
    energy = E.value(u, triangles)
    print(energy)
    energy.backward()
    grad_norm = u.grad.norm()

if __name__ == '__main__':
    demo()
