import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from surface import Mesh
from triangulation_cylinder import generate_cylinder_triangulation



circle_subdivisions = 20  # Number of subdivisions around the circle
height_subdivisions = 10   # Number of subdivisions along the height
height = 6.0            # Actual height of the cylinder

vertices, edges, faces, border = generate_cylinder_triangulation(circle_subdivisions, height_subdivisions, height)
cat_surface_mesh = Mesh(np.array(vertices), np.array(edges), np.array(faces), np.array(border))

cat_surface_mesh.plot_surface()
cat_surface_mesh.gradient_descent(0.25, 5)
cat_surface_mesh.plot_surface()