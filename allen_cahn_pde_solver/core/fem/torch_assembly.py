import torch
from math import comb

# Assemble Pk basis functions on reference triangle
# FOR FUTURE: do for general k
def reference_triangle(k=1):
  vertices = torch.tensor([[0,0],[1,0],[0,1]])
  bar_basis = torch.tensor([[-1,-1,1], [1,0,0], [0,1,0]])
  return bar_basis
    
def assemble_mass_and_stiffness(points: torch.Tensor,
                                triangles: torch.LongTensor,
                                k: int):
  """
  points:   (N,2) float tensor of vertex coords
  triangles:(T,3) long tensor of vertex indices
  returns:
     M: (N,N) sparse tensor
     K: (N,N) sparse tensor
   """

  # 1) Extract triangle vertex coordinates: (T,3,2)
  pts_tri = points[triangles]

  # 2) Compute twice-area via cross of edges (vectorized)
  v01 = pts_tri[:,1] - pts_tri[:,0]   # (T,2)
  v02 = pts_tri[:,2] - pts_tri[:,0]
  cross = v01[:,0]*v02[:,1] - v01[:,1]*v02[:,0]  # (T,)
  area = 0.5 * torch.abs(cross)                   # (T,)

  # 3) Basis matrix
  T = pts_tri.shape[0]
  dtype,device = pts_tri.dtype, pts_tri.device

  ones = torch.ones((T,3,1), dtype=dtype, device=device) # shape (T,3,1)
  A = torch.cat([pts_tri, ones], dim=2) 
  Ainv = torch.linalg.inv(A)   # shape (T,3,3)
  basis_matrix = Ainv # [ [[a1,a2,a3], 
                      #    [b1,b2,b3], 
                      #    [c1,c2,c3]]  ,  .... , ]

  return area, basis_matrix