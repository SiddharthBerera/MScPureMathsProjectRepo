import numpy as np
import matplotlib.pyplot as plt

def gramian(p,q):
    gram_matrix = np.array([[p@p, p@q], [p@q, q@q]])
    return np.linalg.det(gram_matrix)

class Mesh():
    def __init__(self, vertex_list, edge_list, face_list):
        self.vertex_list = vertex_list
        self.edge_list = edge_list
        self.face_list = face_list
    
    
class Triangle():
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.side1 = v1-v2
        self.side2 = v1-v3
        self.area = 0.5*np.sqrt(gramian(self.side1,self.side2))
        self.grad_v1 = (1/ (4*self.area) )*( np.linalg.norm(self.v1-self.v3,2)*( self.v1-self.v2 ) + 
                                        np.linalg.norm(self.v1-self.v2,2)*( self.v1-self.v3 ) -
                                        (self.v1-self.v2)@(self.v1-self.v3)*(2*self.v1-self.v2-self.v3)
                                    )
        self.grad_v2 = (1/ (4*self.area) )*( np.linalg.norm(self.v1-self.v3,2)*( self.v2-self.v1 ) +
                                        (self.v1-self.v2)@(self.v1-self.v3)*(self.v1-self.v3)
                                    )
        self.grad_v3 = (1/ (4*self.area) )*( np.linalg.norm(self.v1-self.v3,2)*( self.v3-self.v1 ) +
                                        (self.v1-self.v2)@(self.v1-self.v3)*(self.v1-self.v2)
                                    )
    
    def plot_2d(self):
        fig, ax = plt.subplots()
        vertices = np.vstack([self.v1, self.v2, self.v3, self.v1])
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-', lw=2)
        ax.scatter(vertices[:-1, 0], vertices[:-1, 1], c='k')
        # Plot the gradient vectors using quiver (with different colors for clarity)
        
        ax.quiver(self.v1[0], self.v1[1], self.grad_v1[0], self.grad_v1[1],
              color='r', angles='xy', scale_units='xy', scale=1, label='grad v1')
        ax.quiver(self.v2[0], self.v2[1], self.grad_v2[0], self.grad_v2[1],
              color='g', angles='xy', scale_units='xy', scale=1, label='grad v2')
        ax.quiver(self.v3[0], self.v3[1], self.grad_v3[0], self.grad_v3[1],
              color='b', angles='xy', scale_units='xy', scale=1, label='grad v3')
    
        ax.set_aspect('equal')
        ax.set_title("2D Triangle and Area Gradients")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        plt.legend()
        plt.show()

    def plot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        # Collect vertices in order and close the loop
        vertices = np.vstack([self.v1, self.v2, self.v3, self.v1])
        ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'k-', lw=2)
        ax.scatter(vertices[:-1, 0], vertices[:-1, 1], vertices[:-1, 2], c='k')
        
        # For 3D plotting, remove the extra keyword arguments
        ax.quiver(self.v1[0], self.v1[1], self.v1[2],
                  self.grad_v1[0], self.grad_v1[1], self.grad_v1[2],
                  color='r', label='grad v1')
        ax.quiver(self.v2[0], self.v2[1], self.v2[2],
                  self.grad_v2[0], self.grad_v2[1], self.grad_v2[2],
                  color='g', label='grad v2')
        ax.quiver(self.v3[0], self.v3[1], self.v3[2],
                  self.grad_v3[0], self.grad_v3[1], self.grad_v3[2],
                  color='b', label='grad v3')
    
        # Setting equal aspect ratio in 3D is more involved.
        # For simplicity, we'll skip it here.
        ax.set_title("3D Triangle and Area Gradients")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.legend()
        plt.show()


        
triangle_1 = Triangle(np.array([0,0,0]),np.array([1,0,0]),np.array([0,1,0]))
print(triangle_1.area) 
triangle_1.plot_3d()

