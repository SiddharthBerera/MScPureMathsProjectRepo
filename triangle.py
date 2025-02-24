import numpy as np
import matplotlib.pyplot as plt

# Ignore in this file, redundant method for computing triangle area in 
# surface.py however might recinclude at later stage to impore modularity
def triangle_area_method2(triangle):
    # extract the lists of sides p,r,q for each face
    p, r, q = triangle[:, 0, :], triangle[:, 1, :], triangle[:, 2, :] # each of shape (F,n)
    rp = r-p
    qp = q-p
    # pointwise mutliply matrices, resulting matrix has rows which when summed correpsond to the dot for sides of a triangle
    rp_dot_rp  = np.sum(rp*rp, axis=1) # shape (F)
    rp_dot_qp = np.sum(rp*qp, axis=1) # shape (F)
    qp_dot_qp = np.sum(qp*qp, axis=1) # shape (F)
    area = 0.5*np.sqrt(rp_dot_rp * qp_dot_qp - qp_dot_qp**2) # shape (F)
    return area

def gramian(p,q):
    return ((p.T@p) * (q.T@q)) - ((q.T@p.T))**2

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
print(triangle_1.grad_v1)
print(triangle_1.grad_v2)
print(triangle_1.grad_v3)
print(triangle_1.area) 
triangle_1.plot_3d()
 