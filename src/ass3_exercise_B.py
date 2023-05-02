import numpy as np

from fem_students_1d import (
    assemble_fe_problem,
    create_fe_space,
    create_mesh,
    create_param_map,
    create_ref_data,
    norm_0,
    norm_1,
)
from utils import eval_func
from utils.plotter import Plotter


def problem_B(
    _: np.ndarray, __: np.ndarray, dNj: np.ndarray, ___: np.ndarray, dNk: np.ndarray
) -> np.ndarray:
    return np.multiply(dNj, dNk)


def problem_L(_: np.ndarray, Nj: np.ndarray, __: np.ndarray) -> np.ndarray:
    return -2 * np.pi * Nj


u_e = lambda x: np.pi * np.power(x, 2) - np.e * x + 1
d_u_e = lambda x: 2 * np.pi * x - np.e

m = 4
p = 2
k = 1
L = 1
neval = 3
spacing_func = lambda i: i * L / m
bc = (1, np.pi - np.e + 1)

brk = np.array([spacing_func(i) for i in range(0, m + 1)])
mesh = create_mesh(brk)
param_map = create_param_map(mesh)
space = create_fe_space(p, k, mesh)
ref_data = create_ref_data(neval, p, True)

A, b = assemble_fe_problem(mesh, space, ref_data, param_map, problem_B, problem_L, bc)

# Solve linear system
u_sol = np.linalg.solve(A, b)
u_coefs = np.zeros(u_sol.shape[0] + 2)
u_coefs[0] = bc[0]
u_coefs[-1] = bc[1]
u_coefs[1:-1] = u_sol

# Increase point evaluations for better resolution
neval = 20
ref_data = create_ref_data(neval, p, True)

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
x_exact = np.linspace(0, L, m * q)
exact = u_e(x_exact)
d_exact = d_u_e(x_exact)
Plotter.plot_results(
    xs_matrix,
    ns_matrix,
    dxns_matrix,
    "ass_exercise_A.pdf",
    result_label="u_h",
    x_exact=x_exact,
    exact_solution=exact,
    d_exact_solution=d_exact,
)

print(norm_0(u_e, u_coefs, mesh, param_map, space, ref_data))
print(norm_1(u_e, d_u_e, u_coefs, mesh, param_map, space, ref_data))
