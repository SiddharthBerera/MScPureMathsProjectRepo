import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Mesh():
    def __init__(self, vertex_list, edge_list, face_list, border_list):
        '''
        Arguments:
        vertex_list: (np.ndarray shape (V,n)) V points in n-d for points of mesh
        edge_list: (np.ndarray shape (E,2)) E edges represented by pairs of vertex indices
        face_list: (np.ndarray shape (F,3)) F faces represented by triples of vertex indices
        border_list: (np.array shape (V)) B indices correpsonding to fixed vertices in vertex_list
        '''
        self.vertex_list = vertex_list.astype(np.float64)
        self.edge_list = edge_list
        self.face_list = face_list
        self.border_list = border_list
        self.grads_list = np.zeros_like(vertex_list, dtype=np.float64) # Shape (V,n)

    @staticmethod
    def grad_triangle(triangles):

        # Extract the lists of sides p,r,q for each face
        # e.g. for p we want for every face (triangle), the first corner, for every coordinate
        p, q, r = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :] # each of shape (F,n)
        
        pr = r-p # shape (F, n)
        pq = q-p # shape (F, n)
        qr = r-q # shape (F, n)

        # pointwise mutliply matrices, resulting matrix has rows which when summed correpsond to the dot for sides of a triangle
        pr_dot_pr = np.sum(pr*pr, axis=1) # shape (F)
        pq_dot_pq = np.sum(pq*pq, axis=1) # shape (F)
        qr_dot_qr = np.sum(qr*qr, axis=1) # shape (F)

        pq_dot_pr = np.sum(pq*pr, axis=1) # shape (F)
        pq_dot_qr = np.sum(pq*qr, axis=1) # shape (F)

        areas = 0.5*np.sqrt(pr_dot_pr * pq_dot_pq - pq_dot_pr**2) # shape (F)
        areas = np.maximum(areas, 0.05) 

        '''
        pr_dot_pr[:, None] broadcasts [p1.r1, ..., pF.rF] to [[p1.r1]*n, ..., [pF.rF]*n]] so that 
        pr_dot_pr[:, None]*pq gives [(p1-q1)*pr_dot_pr1,..., (pF-qF)*pr_dot_prF] via pointwise mult.
        '''
        # e.g. grad_ps[i] stores grad of area of triangles[i] wrt p for i=1,...,no. faces 
        #grad_ps = ( -pr_dot_pr[:, None]*pq - pq_dot_pq[:, None]*pr + pq_dot_pr[:, None]*(pr+pq) ) / ( 4 * areas[:, None]) # shape (F,n)
        grad_ps = ( qr_dot_qr[:, None]*(p-q) - pq_dot_qr[:, None]*(q-r)) / ( 4 * areas[:, None]) # shape (F,n)
        grad_qs = ( pr_dot_pr[:, None]*(q-p) - pq_dot_pr[:, None]*(r-p)) / ( 4 * areas[:, None]) # shape (F,n)
        grad_rs = ( pq_dot_pq[:, None]*(r-p) - pq_dot_pr[:, None]*(q-p)) / ( 4 * areas[:, None]) # shape (F,n)

        # We want all_grads[i] to be the matrix [grad_ps[i], grad_qs[i], grad_rs[i]]
        all_grads =  np.stack([grad_ps, grad_qs, grad_rs], axis=1) # Shape (F,3,n)
        return all_grads
    
    def zero_grads(self):
        self.grads_list[:,:] = 0

    def compute_mesh_grads(self):
        # use face list as a mask to get all triples of nd coords characterising each of F triangles
        triangles = self.vertex_list[self.face_list] # triangles of shape (F,3,n)

        # use vectorised grad_triangle function to compute for each of F triangles, the 3 nd grad vectors 
        grads_unaggregated = self.grad_triangle(triangles)

        # np.add.at(self.grads_list, face[i], grads[i]) will add in positions face[i] of grad list grads[i]
        # since a vectorised function works for a list of face[i]s and a list of grad[i]s
        np.add.at(self.grads_list, self.face_list, grads_unaggregated)

        # ensure boundary stays fixed
        self.grads_list[self.border_list, :] = 0

    def gradient_descent(self, stepsize, iterations):
        for i in range(iterations):
            # ensure we start with a zeroed grad list of 
            self.zero_grads()
            # compute the grads 
            self.compute_mesh_grads()
            # perfrom the gradient descent step
            self.vertex_list = self.vertex_list - stepsize*self.grads_list

    def plot_surface(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Build a list of triangles (each a list of three vertex coordinates) for plotting.
        triangles = self.vertex_list[self.face_list]
        
        # Plot the surface as a blue semi-transparent mesh.
        poly_collection = Poly3DCollection(triangles, facecolors='blue', edgecolors='k', alpha=0.1)
        ax.add_collection3d(poly_collection)
        
        # create a mask for differentiating between fixed and non-fixed points for plotting
        mask_border = np.zeros(self.vertex_list.shape[0], dtype=bool)
        mask_border[self.border_list] = True
        mask_non_border = ~mask_border

        # Plot the non-fixed vertices as red points
        ax.scatter(self.vertex_list[mask_non_border, 0], self.vertex_list[mask_non_border, 1], 
                   self.vertex_list[mask_non_border, 2], color='red', s=50, label='Vertices')
        
        # plot the fixed vertices as green points
        ax.scatter(self.vertex_list[self.border_list, 0], self.vertex_list[self.border_list, 1],
                   self.vertex_list[self.border_list, 2], color='green', s=50, label='Boundary')
        
        # Set axis limits with a little margin.
        margin = 0.1
        x_min, x_max = self.vertex_list[:, 0].min() - margin, self.vertex_list[:, 0].max() + margin
        y_min, y_max = self.vertex_list[:, 1].min() - margin, self.vertex_list[:, 1].max() + margin
        z_min, z_max = self.vertex_list[:, 2].min() - margin, self.vertex_list[:, 2].max() + margin
        #ax.set_xlim(x_min, x_max)
        #ax.set_ylim(y_min, y_max)
        #ax.set_zlim(z_min, z_max)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("Surface after Gradient Descent")
        plt.legend()
        plt.show()

    
    
# --- Example usage ---
if __name__ == "__main__":
    mesh_grid1 = np.array([[0,0,0],[1,0,0],[0,1,0], [0,0,1]])
    edge_list1 = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    face_list1 = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
    fixed = np.array([False, False, False, False])

    mesh1 = Mesh(mesh_grid1, edge_list1, face_list1, fixed)
    mesh1.compute_mesh_grads()
    l = mesh1.grads_list
    print(l)
    print(np.sum(l,axis=1))

    mesh1.plot_surface()
    print('VERTEX')
    print(mesh1.vertex_list)
    mesh1.gradient_descent(1, 1)
    mesh1.plot_surface()
    print(mesh1.vertex_list)



