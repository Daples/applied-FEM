from utils import read_mat
from fem.bi_dim_geometric_map import BidimensionalGeometricMap
from fem.bi_dim_reference_data import BidimensionalReferenceData
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
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


fe_geometry, fe_space = read_mat("data/star3.mat")

ref_data = BidimensionalReferenceData(20, [2, 2], False)
geom_map = BidimensionalGeometricMap(fe_geometry, ref_data)

n = fe_space.n
coefs = np.zeros((n))
coefs[2] = 1

x, y = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))


grid_x = np.zeros((20, 20))
grid_y = np.zeros((20, 20))
grid_u = np.zeros((20, 20))
grid_dx = np.zeros((20, 20))
grid_dy = np.zeros((20, 20))

xs = np.zeros((0,))
ys = np.zeros((0,))
us = np.zeros((0,))
dxs = np.zeros((0,))
dys = np.zeros((0,))
for current_element in range(fe_geometry.m):
    support_extractors = fe_space.support_extractors[current_element]
    u, dx, dy = eval_func(
        ref_data, support_extractors, geom_map, current_element, coefs, 20**2
    )
    x = geom_map.map[:, 0, current_element]
    y = geom_map.map[:, 1, current_element]

    for i in range(20):
        for j in range(20):
            I = j * 20 + i
            grid_x[i, j] = x[I]
            grid_y[i, j] = y[I]
            grid_u[i, j] = u[0, I]
            grid_dx[i, j] = dx[0, I]
            grid_dy[i, j] = dy[0, I]

    xs = np.hstack((xs, grid_x.ravel()))
    ys = np.hstack((ys, grid_y.ravel()))
    us = np.hstack((us, grid_u.ravel()))
    dxs = np.hstack((dxs, grid_dx.ravel()))
    dys = np.hstack((dys, grid_dy.ravel()))


fig, axs = plt.subplots(2, 2, figsize=(15, 10), subplot_kw=dict(projection="3d"))

surf = axs[0, 0].plot_trisurf(xs, ys, us, cmap="jet", linewidth=0)
fig.colorbar(surf)
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("y")
axs[0, 0].set_zlabel("z")
axs[0, 0].xaxis.set_major_locator(MaxNLocator(5))
axs[0, 0].yaxis.set_major_locator(MaxNLocator(6))
axs[0, 0].zaxis.set_major_locator(MaxNLocator(5))
axs[0, 0].set_title("$N$")

surf = axs[0, 1].plot_trisurf(xs, ys, dxs, cmap="jet", linewidth=0)
fig.colorbar(surf)
axs[0, 1].xaxis.set_major_locator(MaxNLocator(5))
axs[0, 1].yaxis.set_major_locator(MaxNLocator(6))
axs[0, 1].zaxis.set_major_locator(MaxNLocator(5))
axs[0, 1].set_title("$N_x$")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("y")
axs[0, 1].set_zlabel("z")

surf = axs[1, 1].plot_trisurf(xs, ys, dys, cmap="jet", linewidth=0)
fig.colorbar(surf)
axs[1, 1].xaxis.set_major_locator(MaxNLocator(5))
axs[1, 1].yaxis.set_major_locator(MaxNLocator(6))
axs[1, 1].zaxis.set_major_locator(MaxNLocator(5))
axs[1, 1].set_title("$N_y$")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("y")
axs[1, 1].set_zlabel("z")

fig.tight_layout()

plt.show()  # or:
