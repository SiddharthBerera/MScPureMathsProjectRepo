# triangulation_cylinder.py

import math

def generate_cylinder_triangulation(circle_subdivisions, height_subdivisions, height):
    """
    Generates a triangulation for the lateral surface of a cylinder using 0-based indexing.
    
    Parameters:
        circle_subdivisions (int): Number of subdivisions along the circular direction.
                                   (Must be at least 3 for a proper closed circle.)
        height_subdivisions (int): Number of subdivisions along the height.
                                   (Must be at least 2 to have a top and bottom row.)
        height (float): The actual height of the cylinder.
        
    Returns:
        vertices (list of tuples): List of vertex coordinates (x, y, z).
                                   The first entry corresponds to vertex index 0.
        faces (list of lists): List of triangle faces, each specified as a list of 3 vertex indices.
                               The indices are 0-based.
        border (list of int): List of vertex indices (0-based) on the boundary (i.e. bottom and top rows).
    """
    # Assume a unit radius for the cylinder
    r = 1.0
    vertices = []
    
    # Generate vertices: iterate over each height level and circular subdivision.
    for i in range(height_subdivisions):
        # Evenly space z between 0 and height.
        z = height * i / (height_subdivisions - 1)
        for j in range(circle_subdivisions):
            theta = 2 * math.pi * j / circle_subdivisions
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            vertices.append((x, y, z))
    
    faces = []
    # Build the triangulation for each quad cell.
    # Vertices are arranged in a grid: for each cell defined by height index i and circular index j.
    for i in range(height_subdivisions - 1):
        for j in range(circle_subdivisions):
            next_j = (j + 1) % circle_subdivisions

            # Compute the 0-based indices for the vertices:
            bottom_left  = i * circle_subdivisions + j
            bottom_right = i * circle_subdivisions + next_j
            top_left     = (i + 1) * circle_subdivisions + j
            top_right    = (i + 1) * circle_subdivisions + next_j

            # Create two triangles for the quad cell.
            faces.append([bottom_left, top_left, top_right])
            faces.append([bottom_left, top_right, bottom_right])
    
    # Determine the border vertices.
    # The border corresponds to the bottom row (first height level) and the top row (last height level).
    border = list(range(0, circle_subdivisions))  # Bottom row indices
    border += list(range((height_subdivisions - 1) * circle_subdivisions, height_subdivisions * circle_subdivisions))  # Top row indices
    
    return vertices, faces, border

# --- Example usage ---
if __name__ == "__main__":
    circle_subdivisions = 4  # Number of subdivisions around the circle
    height_subdivisions = 5   # Number of subdivisions along the height
    height = 5.0            # Actual height of the cylinder
    
    vertices, faces, border = generate_cylinder_triangulation(circle_subdivisions, height_subdivisions, height)
    
    print("Vertices (first 5 shown):")
    for idx, vertex in enumerate(vertices[:5]):
        print(f"Index {idx}: {vertex}")
    
    print("\nFirst 5 faces:")
    for face in faces[:5]:
        print(face)
    
    print("\nBorder vertex indices:")
    print(border)
