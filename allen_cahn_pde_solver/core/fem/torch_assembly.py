import torch
from math import comb

# Assemble Pk basis functions on reference triangle
# FOR FUTURE: do for general k
def reference_triangle(k=1):
  vertices = torch.tensor([[0,0],[1,0],[0,1]])
  bar_basis = torch.tensor([[-1,-1,1], [1,0,0], [0,1,0]])
  return bar_basis
    
def assemble_basis(points: torch.Tensor,
                   triangles: torch.LongTensor,
                   k: int):
  """
  points:   (N,2) float tensor of vertex coords
  triangles:(T,3) long tensor of vertex indices
  returns:
    area          : (T,)           area of each triangle
    basis_matrix  : (T,3,3)
                    basis functions λ_i
                    rows = [a_i, b_i, c_i] so that
                            λ_i(x,y) = a_i x + b_i y + c_i
  """

  # 1) Extract triangle vertex coordinates: (T,3,2)
  pts_tri = points[triangles]

  # 2) Compute area via cross of edges (vectorized)
  v01 = pts_tri[:,1] - pts_tri[:,0]   # (T,2)
  v02 = pts_tri[:,2] - pts_tri[:,0]
  cross = v01[:,0]*v02[:,1] - v01[:,1]*v02[:,0]  # (T,)
  area = 0.5 * torch.abs(cross)                  # (T,)

  # 3) Basis matrix
  ones = torch.ones((pts_tri.shape[0],3,1), 
                    dtype=pts_tri.dtype, 
                    device=pts_tri.device) # shape (T,3,1)
  A = torch.cat([pts_tri, ones], dim=2) 
  Ainv = torch.linalg.inv(A)   # shape (T,3,3)
  basis_matrix = Ainv.transpose(1,2) # [ [[a1,b1,c1], 
                                     #    [a2,b2,c2], 
                                     #    [a3,b3,c3]]  ,  .... , ]

  return area, basis_matrix