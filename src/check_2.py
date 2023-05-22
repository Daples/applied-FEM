from utils import read_mat
from fem.create_geometric_map import create_geometric_map
from fem.create_ref_data import create_ref_data
import numpy as np 
import matplotlib.pyplot as plt
from fem.assembler import Assembler
from matplotlib.ticker import MaxNLocator

def eval_func(ref_data, support_extractors, geom_map, current_element, coefs, n):
    ns = np.zeros((1, n))
    dxns = np.zeros((1, n))
    dyns = np.zeros((1, n))

    for i, j in enumerate(support_extractors.supported_bases):
        ej_i = support_extractors.extraction_coefficients[i, :]
        ns += coefs[j] * ej_i.dot(ref_data.reference_basis)
        dxns += coefs[j] * ej_i.dot(
            np.multiply(
                geom_map.imap_derivatives[:, 0, current_element],
                ref_data.reference_basis_derivatives[:, :, 0],
            )
            + np.multiply(
                geom_map.imap_derivatives[:, 1, current_element],
                ref_data.reference_basis_derivatives[:, :, 1],
            )
        )
        dyns += coefs[j] * ej_i.dot(
            np.multiply(
                geom_map.imap_derivatives[:, 2, current_element],
                ref_data.reference_basis_derivatives[:, :, 0],
            )
            + np.multiply(
                geom_map.imap_derivatives[:, 3, current_element],
                ref_data.reference_basis_derivatives[:, :, 1],
            )
        )

    return ns, dxns, dyns


problem_B = lambda x, Nj, dxNj, dyNj, Nk, dxNk, dyNk: np.multiply(dxNj, dxNk) + np.multiply(dyNj, dyNk)
problem_L = lambda x, Nj, dxNj, dyNj: Nj 

fe_geometry, fe_space = read_mat("data/distressed_robotD.mat")

ref_data = create_ref_data(3, [2, 2], True)
geom_map = create_geometric_map(fe_geometry, ref_data)

A, b, idxs = Assembler.two_dimensional(fe_space,
        ref_data,
        geom_map,
        fe_geometry,
        problem_B,
        problem_L)


u = np.linalg.solve(A,b)
u_sol = np.zeros(fe_space.n)
u_sol[idxs] = u

n_plot = 20

ref_data = create_ref_data(n_plot, [2, 2], False)
geom_map = create_geometric_map(fe_geometry, ref_data)

plot_grid_x = np.zeros((n_plot,n_plot))
plot_grid_y = np.zeros((n_plot,n_plot))
plot_grid_u = np.zeros((n_plot,n_plot))
plt.figure()

for current_element in range(fe_geometry.m):
    support_extractors = fe_space.support_extractors[current_element]
    u, dx, dy= eval_func(ref_data, support_extractors, geom_map, current_element, u_sol, n_plot ** 2)
    x = geom_map.map[:,0,current_element]
    y = geom_map.map[:,1,current_element]

    for i in range(n_plot):
        for j in range(n_plot):
            I = j * n_plot + i
            plot_grid_x[i, j] = x[I]
            plot_grid_y[i, j] = y[I]
            plot_grid_u[i, j] = u[0,I]


    plt.contourf(plot_grid_x, plot_grid_y, plot_grid_u, cmap ='viridis', vmin=0, vmax=10)
    plt.clim(0, 10) 
    plt.title('Heatmap of the exact solution')


plt.colorbar()    
plt.show()


