import numpy as np
import math
import torch
import meshzoo

from .domain import Domain


class TriangleDomain(Domain):
    def __init__(self, vertex_list: torch.tensor, 
                 edge_list: torch.tensor, 
                 face_list: torch.tensor, 
                 border_list: torch.tensor):
        '''
        Arguments:
        vertex_list: (np.ndarray shape (V,n)) V points in n-d for points of mesh
        edge_list: (np.ndarray shape (E,2)) E edges represented by pairs of vertex indices
        face_list: (np.ndarray shape (F,3)) F faces represented by triples of vertex indices
        border_list: (np.array shape (V)) B indices correpsonding to fixed vertices in vertex_list
        '''
        self.vertex_list = vertex_list
        self.edge_list = edge_list
        self.face_list = face_list
        self.border_list = border_list

    def mesh(self, h: float):
        pass
    
    def boundary_nodes(self, mesh):
        pass