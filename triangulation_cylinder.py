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
        border (list of int): List of vertex indices (0-based) on the boundary (i.e., bottom and top rows).
        edges (list of lists): List of unique edges, each specified as a sorted list [i, j] of vertex indices.
    """
    r = 1.0
    vertices = []
    
    # Generate vertices for each height level and circular subdivision.
    for i in range(height_subdivisions):
        z = height * i / (height_subdivisions - 1)
        for j in range(circle_subdivisions):
            theta = 2 * math.pi * j / circle_subdivisions
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            vertices.append((x, y, z))
    
    faces = []
    # Create faces by splitting each quadrilateral cell into two triangles.
    for i in range(height_subdivisions - 1):
        for j in range(circle_subdivisions):
            next_j = (j + 1) % circle_subdivisions

            bottom_left  = i * circle_subdivisions + j
            bottom_right = i * circle_subdivisions + next_j
            top_left     = (i + 1) * circle_subdivisions + j
            top_right    = (i + 1) * circle_subdivisions + next_j

            faces.append([bottom_left, top_left, top_right])
            faces.append([bottom_left, top_right, bottom_right])
    
    # Determine the border vertices (bottom and top rows).
    border = list(range(0, circle_subdivisions))  # Bottom row.
    offset = (height_subdivisions - 1) * circle_subdivisions
    border += list(range(offset, offset + circle_subdivisions))  # Top row.
    
    # Build the edge list from the faces.
    # For each face, add its three edges (as sorted tuples) to a set to avoid duplicates.
    edge_set = set()
    for face in faces:
        # Each face is a list of three vertices: [v0, v1, v2]
        edges_in_face = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for edge in edges_in_face:
            # Store the edge with the smaller index first.
            sorted_edge = tuple(sorted(edge))
            edge_set.add(sorted_edge)
    
    # Convert the set of edges to a sorted list.
    edges = [list(edge) for edge in sorted(edge_set)]
    
    return vertices, edges, faces, border

# --- Example usage ---
if __name__ == "__main__":
    circle_subdivisions = 4  # e.g., 12 subdivisions around the circle.
    height_subdivisions = 5   # e.g., 5 subdivisions along the height.
    height = 5              # Cylinder height.
    
    vertices, edges, faces, border  = generate_cylinder_triangulation(circle_subdivisions, height_subdivisions, height)
    
    print("Vertices (first 5):")
    for i, v in enumerate(vertices[:5]):
        print(f"Vertex {i}: {v}")
    
    print("\nFirst 5 faces:")
    for face in faces[:5]:
        print(face)
    
    print("\nBorder vertex indices:")
    print(border)
    
    print("\nEdge list (first 10 edges):")
    for edge in edges[:10]:
        print(edge)
