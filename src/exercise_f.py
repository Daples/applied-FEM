import numpy as np

from fem.mesh import Mesh
from fem.param_map import ParamMap
from fem.reference_data import ReferenceData
from fem.space import Space
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
mesh = Mesh(brk)
param_map = ParamMap(mesh)
space = Space(p, k, mesh)
ref_data = ReferenceData(neval, p, False)

m = mesh.elements.shape[1]
q = ref_data.evaluation_points.shape[0]
xs_matrix = np.zeros((m, q))
ns_matrix = np.zeros_like(xs_matrix)
dxns_matrix = np.zeros_like(ns_matrix)

for current_element in range(m):
    element = mesh.elements[:, current_element]
    xs, ns, dxns = eval_func(
        current_element, coefs, element, param_map, space, ref_data
    )
    xs_matrix[current_element, :] = xs
    ns_matrix[current_element, :] = ns
    dxns_matrix[current_element, :] = dxns

Plotter.plot_results(
    xs_matrix, ns_matrix, dxns_matrix, "exercise_f.pdf", result_label="u_h"
)
