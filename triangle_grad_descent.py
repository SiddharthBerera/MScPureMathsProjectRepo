import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from copy import deepcopy

from surface import Mesh
from triangulation_cylinder import generate_cylinder_triangulation



mesh_grid1 = np.array([[0,0,0],[1,0,0],[0,1,0]])
edge_list1 = np.array([[0,1],[0,2],[1,2]])
face_list1 = np.array([[0,1,2]])
border_list1 = np.array([0,1])

#vertices, edges, faces, border = generate_cylinder_triangulation(circle_subdivisions, height_subdivisions, height)
cat_surface_mesh = Mesh(mesh_grid1, edge_list1, face_list1, border_list1)
cat_surface_mesh.plot_surface()
cat_surface_mesh.gradient_descent(0.25, 10)
cat_surface_mesh.plot_surface()
