import torch
import math
from allen_cahn_pde_solver.core.fem.barycentric import bary_integral

class AllenCahnEnergy:
    """
    Assemble and evaluate the Allen-Cahn Energy
    E_e(u) = 0.5e int_Omega |grad_u|^2 + (1-u^2)^2/(4e)    
    for a P1 finite-element field on a triangular mesh.
    ----------
    area: PyTorch tensor shape (T,) of each triangles area
    basis_matrix: (T,N,N) PyTorch tensors of basis matrix for each triangle
    eps : interface thickness
    """

    def __init__(self, areas: torch.Tensor, basis_matrices: torch.Tensor, eps: float):
        self.areas = areas
        self.basis_matrices = basis_matrices
        self.a_b = self.basis_matrices[:, :, 0:2].to(areas.dtype)

        self.eps = eps

        # degree-4 barycentric integrals (scalars on the same device / dtype)
        dev, dt = areas.device, areas.dtype
        self.I400  = bary_integral(4,0,0, dtype=dt, device=dev)   # ∫ λ_i⁴
        self.I310  = bary_integral(3,1,0, dtype=dt, device=dev)   # ∫ λ_i³ λ_j
        self.I220  = bary_integral(2,2,0, dtype=dt, device=dev)   # ∫ λ_i² λ_j²
        self.I211 = bary_integral(2,1,1, dtype=dt, device=dev)   # ∫ λ_i² λ_j λ_k
        self.I200 = bary_integral(2,0,0, dtype=dt, device=dev)   # ∫ λ_i²
        self.I110 = bary_integral(1,1,0, dtype=dt, device=dev)   # ∫ λ_i λ_j

    # ---- potential energy on each triangle ----------------------------
    def _potential_per_triangle(self, u1, u2, u3):
        """
        Compute int_triangle (1-u^2)^2 for every triangle in a vectorised way
        ----------
        Arguments are (T,) tensors.
        ----------
        Returns a (T,) tensors.
        """

        potential = self.areas * ( 1
        + self.I400 * ( u1**4 + u2**4 + u3**4 ) 
        - 2 * self.I200 * ( u1**2 + u2**2 + u3**2 )
        - 4 * self.I110 * ( u1*u2 + u1*u3 + u2*u3 )
        + 6 * self.I220 * ( (u1*u2)**2 + (u1*u3)**2 + (u2*u3)**2 )
        + 4 * self.I310 * ( u1*u2**3 + u1**3*u2 + u1*u3**3 + u1**3*u3 + u2*u3**3 + u2**3*u3)
        + 12 * self.I211 * (u1**2*u2*u3 + u1*u2**2*u3 + u1*u2*u3**2 ) ) / (4*self.eps)

        return potential   # shape (T,)
    
    def _dirichlet_energy_per_triangle(self, u1, u2, u3):
        """
        Compute int_triangle |grad u|^2 for every triangle in a vectorised way
        ----------
        Arguments are (T,) tensors.
        ----------
        Returns a (T,) tensors.
        """
        # gradients: ∑_i u_i * (a_i,b_i)
        grads = torch.einsum('t i d, t i -> t d', self.a_b, torch.stack((u1, u2, u3), dim=1))  # (T,2)
        dirich = 0.5 * self.eps * (grads.square().sum(dim=1) * self.areas)  # (T,)
        return dirich  # shape (T,)


    # ---- full energy --------------------------------------------------
    def value(self, u: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
        """
        Compute the full Energy
        ----------
        u          : (N,) nodal values
        triangles  : (T,3) index tensor
        ----------
        E(u)  : scalar tensor (same dtype/device as inputs)
        """
        #  nodal values per triangle
        u_tri = u[triangles]                  # (T,3)
        u1, u2, u3 = u_tri.unbind(dim=1)      # each (T,)

        # ---- potential part ------------------------------------------
        pot = self._potential_per_triangle(u1, u2, u3).sum()

        # ---- Dirichlet part ------------------------------------------
        dirich = self._dirichlet_energy_per_triangle(u1, u2, u3).sum()
        
        energy = pot + dirich
        return energy
    
    
    @staticmethod
    def triangle_energy(u_tri, vertices, params):
        """
        Compute Allen-Cahn energy for a single triangle.
        u_tri: (3,) nodal values
        area: scalar
        basis_matrix: (3,2)
        eps: float
        """

        vertices = vertices.to(dtype=torch.float64)
        eps = params['eps']
        
        I400 = 1/15
        I310 = 1/60
        I220 = 1/90
        I211 = 1/180
        I200 = 1/6
        I110 = 1/12
        

        v0, v1, v2 = vertices
        e1, e2 = v1 - v0, v2 - v0
        area = 0.5 * torch.abs(e1[0] * e2[1] - e1[1] * e2[0])
        
        u1, u2, u3 = u_tri


        # Potential part (copy from _potential_per_triangle, but for one triangle)
        pot = area * ( 1
        + I400 * ( u1**4 + u2**4 + u3**4 ) 
        - 2 * I200 * ( u1**2 + u2**2 + u3**2 )
        - 4 * I110 * ( u1*u2 + u1*u3 + u2*u3 )
        + 6 * I220 * ( (u1*u2)**2 + (u1*u3)**2 + (u2*u3)**2 )
        + 4 * I310 * ( u1*u2**3 + u1**3*u2 + u1*u3**3 + u1**3*u3 + u2*u3**3 + u2**3*u3)
        + 12 * I211 * (u1**2*u2*u3 + u1*u2**2*u3 + u1*u2*u3**2 ) ) / (4*eps)

        # Dirichlet part
        ones = torch.ones((3,1), dtype=vertices.dtype, device=vertices.device) # shape (3,1)
        A = torch.cat([vertices, ones], dim=1) 
        print(torch.det(A))
        Ainv = torch.linalg.inv(A)   # shape (3,3)
        basis_matrix = Ainv.T
        basis_matrix_xy = basis_matrix[:, :2]
        grads = torch.einsum('i d, i -> d', basis_matrix_xy, u_tri)
        dirich = 0.5 * eps * grads.square().sum() * area

        return pot + dirich