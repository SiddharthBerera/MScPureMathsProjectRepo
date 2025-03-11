import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from copy import deepcopy

from surface import Mesh
from triangulation_cylinder import generate_cylinder_triangulation



circle_subdivisions = 10  # Number of subdivisions around the circle
height_subdivisions = 10   # Number of subdivisions along the height
height = 0.8          # Actual height of the cylinder

vertices, edges, faces, border = generate_cylinder_triangulation(circle_subdivisions, height_subdivisions, height)
cat_surface_mesh = Mesh(np.array(vertices), np.array(edges), np.array(faces), np.array(border))
cat_surface_mesh.plot_surface()
print(cat_surface_mesh.vertex_list[cat_surface_mesh.border_list])
cat_surface_mesh.gradient_descent(0.05, 100)
print(cat_surface_mesh.vertex_list[cat_surface_mesh.border_list])
cat_surface_mesh.plot_surface()
