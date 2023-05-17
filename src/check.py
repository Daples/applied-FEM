from utils import read_mat
from fem.create_geometric_map import create_geometric_map
from fem.create_ref_data import create_ref_data
import numpy as np 
import matplotlib.pyplot as plt

def eval_func(ref_data, support_extractors, geom_map, current_element, coefs, n):

    ns = np.zeros((1, n))
    dxns = np.zeros((1, n))
    dyns = np.zeros((1, n))

    for i, j in enumerate(support_extractors.supported_bases):
        ej_i = support_extractors.extraction_coefficients[i, :]
        ns += coefs[j] * ej_i.dot(ref_data.reference_basis)
        dxns += (
            coefs[j]
            * ej_i.dot(np.multiply(geom_map.imap_derivatives[:,0,current_element], ref_data.reference_basis_derivatives[:,:,0])
                       +np.multiply(geom_map.imap_derivatives[:,1,current_element], ref_data.reference_basis_derivatives[:,:,1]))
        )
        dyns += (
            coefs[j]
            * ej_i.dot(np.multiply(geom_map.imap_derivatives[:,2,current_element], ref_data.reference_basis_derivatives[:,:,0])
                       +np.multiply(geom_map.imap_derivatives[:,3,current_element], ref_data.reference_basis_derivatives[:,:,1]))
        )

    return ns, dxns, dyns


fe_geometry, fe_space = read_mat("data/star3.mat")

ref_data = create_ref_data(20, [2, 2], False)
geom_map = create_geometric_map(fe_geometry, ref_data)

n = fe_space.n
coefs = np.zeros((n))
coefs[2] = 1

x, y = np.meshgrid(np.linspace(0,1,3),np.linspace(0,1,3))


plot_grid_x = np.zeros((20,20))
plot_grid_y = np.zeros((20,20))
plot_grid_u = np.zeros((20,20))
plt.figure()

for current_element in range(fe_geometry.m):
    support_extractors = fe_space.support_extractors[current_element]
    u, dx, dy= eval_func(ref_data, support_extractors, geom_map, current_element, coefs, 20 ** 2)
    x = geom_map.map[:,0,current_element]
    y = geom_map.map[:,1,current_element]

    for i in range(20):
        for j in range(20):
            I = j * 20 + i
            plot_grid_x[i, j] = x[I]
            plot_grid_y[i, j] = y[I]
            plot_grid_u[i, j] = dy[0,I]


    plt.contourf(plot_grid_x, plot_grid_y, plot_grid_u, cmap ='viridis')
    plt.title('Heatmap of the exact solution')


plt.colorbar()    
plt.show()




