import numpy as np

from fem.assembler import Assembler
from fem.mesh import Mesh
from fem.param_map import ParametricMap
from fem.reference_data import ReferenceData
from fem.space import Space
from utils import eval_func
from utils.plotter import Plotter

problem_B = lambda x, Nj, dNj, Nk, dNk: np.multiply(dNj, dNk)
problem_L = lambda x, Nj, dNj: (np.pi**2) * np.multiply(Nj, np.sin(np.pi * x))

m = 4
p = 2
k = 0
L = 2
neval = 3
spacing_func = lambda i: i * L / m
bc = (0.0, 1.0)

# Initialize problem and assemble matrices
brk = np.array([spacing_func(i) for i in range(0, m + 1)])
mesh = Mesh(brk)
param_map = ParametricMap(mesh)
space = Space(p, k, mesh)
ref_data = ReferenceData(neval, p, True)
A, b = Assembler.one_dimensional(
    mesh, space, ref_data, param_map, problem_B, problem_L, bc
)

# Save system matrices
np.savetxt("0/A.txt", A, delimiter=",", fmt="%1.14f")
np.savetxt("0/b.txt", b, delimiter=",", fmt="%1.14f")

# Solve linear system
u_sol = np.linalg.solve(A, b)
u_coefs = np.zeros(u_sol.shape[0] + 2)
u_coefs[0] = bc[0]
u_coefs[-1] = bc[1]
u_coefs[1:-1] = u_sol

# Increase point evaluations for better resolution
neval = 20
ref_data = ReferenceData(neval, p, True)

# Recover solution
m = mesh.elements.shape[1]
q = ref_data.evaluation_points.shape[0]
xs_matrix = np.zeros((m, q))
ns_matrix = np.zeros_like(xs_matrix)
dxns_matrix = np.zeros_like(ns_matrix)

for current_element in range(mesh.elements.shape[1]):
    element = mesh.elements[:, current_element]
    xs, ns, dxns = eval_func(
        current_element, u_coefs, element, param_map, space, ref_data
    )
    xs_matrix[current_element, :] = xs
    ns_matrix[current_element, :] = ns
    dxns_matrix[current_element, :] = dxns

# Plot
Plotter.plot_results(
    xs_matrix, ns_matrix, dxns_matrix, "exercise_n.pdf", result_label="u_h"
)
