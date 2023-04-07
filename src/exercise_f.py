import numpy as np

from fem_students_1d import (
    create_fe_space,
    create_mesh,
    create_param_map,
    create_ref_data,
)
from utils import eval_func
from utils.plotter import Plotter

coefs = np.loadtxt("data/coefficients.txt")

m = 5
p = 2
k = 1
L = 2
neval = 20
spacing_func = lambda i: i**2 * L / m**2

brk = np.array([spacing_func(i) for i in range(0, m + 1)])
mesh = create_mesh(brk)
param_map = create_param_map(mesh)
space = create_fe_space(p, k, mesh)
ref_data = create_ref_data(neval, p, False)

m = mesh.elements.shape[1]
q = ref_data.evaluation_points.shape[0]
xs_matrix = np.zeros((m, q))
ns_matrix = np.zeros_like(xs_matrix)
dxns_matrix = np.zeros_like(ns_matrix)

for l in range(m):
    element = mesh.elements[:, l]
    xs, ns, dxns = eval_func(l, coefs, element, param_map, space, ref_data)
    xs_matrix[l, :] = xs
    ns_matrix[l, :] = ns
    dxns_matrix[l, :] = dxns

Plotter.plot_results(xs_matrix, ns_matrix, dxns_matrix, "exercise_f.pdf")
