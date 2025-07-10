import numpy as np
import math
import torch
import meshzoo

from .domain import Domain


class TriangleDomain(Domain):
    def __init__(self, vertices=((0,0), (1,0), (0,1))):
        self.vertices = np.asarray(vertices)
    
    def mesh(self, h):
        """
        Always return a mesh of the UNIT right triangle with max-edge≈h.
        """
        # 1) number of subdivisions per edge
        n = math.ceil(1.0 / h)

        # 2) get reference triangle mesh on (0,0),(1,0),(0,1)
        ref_pts, ref_cells = meshzoo.triangle(n)
        # Now ref_pts.shape == (N_pts, 2), ref_cells.shape == (N_tri, 3)

        # 3) convert to tensors
        pts = torch.tensor(ref_pts, dtype=torch.float64)    # (N_pts, 2)
        cells = torch.tensor(ref_cells, dtype=torch.long)   # (N_tri, 3)

        return pts, cells
    
    def boundary_nodes(self, mesh):
        return mesh.boundary_nodes()