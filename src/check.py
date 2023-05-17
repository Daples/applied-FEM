from utils import read_mat
from fem.create_geometric_map import create_geometric_map
from fem.create_ref_data import create_ref_data
import numpy as np 

def eval_func(ref_data, support_extractors, geom_map, current_element, coefs, n):

    ns = np.zeros((1, n))
    dxns = np.zeros((1, n))
    dyns = np.zeros((1, n))

    for i, j in enumerate(support_extractors.supported_bases[current_element]):
        ej_i = support_extractors.extraction_coefficients[i, :]
        ns += coefs[j] * ej_i.dot(ref_data.reference_basis)
        dxns += (
            coefs[j]
            * ej_i.dot(np.multiply(geom_map.imap_derivatives[:,0,current_element], ref_data.reference_basis_derivatives[:,:,0])
                       +np.multiply(geom_map.imap_derivatives[:,1,current_element], ref_data.reference_basis_derivatives[:,:,1]))
        )
        dyns += (
            coefs[j]
            * ej_i.dot(np.multiply(geom_map.imap_derivatives[:,1,current_element], ref_data.reference_basis_derivatives[:,:,0])
                       +np.multiply(geom_map.imap_derivatives[:,3,current_element], ref_data.reference_basis_derivatives[:,:,1]))
        )

    return ns, dxns, dyns


fe_geometry, fe_space = read_mat("data/star3.mat")

ref_data = create_ref_data(3, [2, 2], True)
geom_map = create_geometric_map(fe_geometry, ref_data)

n = fe_space.n
coefs = np.zeros((n))
coefs[6] = 1

x, y = np.meshgrid(np.linspace(0,1,3),np.linspace(0,1,3))


plot_grid_x = np.zeros((3,3))
plot_grid_y = np.zeros((3,3))
plot_grid_u = np.zeros((3,3))

for current_element in range(fe_geometry.m):
    support_extractors = fe_space.support_extractors[current_element]
    print(eval_func(ref_data, support_extractors, geom_map, current_element, coefs, 9))

