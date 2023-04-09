import numpy as np

from fem_students_1d import (
    assemble_fe_problem,
    create_fe_space,
    create_mesh,
    create_param_map,
    create_ref_data,
)
from utils import eval_func
from utils.plotter import Plotter


def problem_B(
    x: np.ndarray, Nj: np.ndarray, dNj: np.ndarray, Nk: np.ndarray, dNk: np.ndarray
) -> np.ndarray:
    return np.multiply(
        1 - 0.4 * np.cos(np.pi * x), np.multiply(dNj, dNk)
    ) + np.multiply(Nj, Nk)


def problem_L(x: np.ndarray, Nj: np.ndarray, _: np.ndarray) -> np.ndarray:
    fd = np.zeros(x.shape[0])
    for z in range(x.shape[0]):
        if x[z] < 1:
            fd[z] = 1
        else:
            fd[z] = -1
    return (np.pi**2) * np.multiply(fd, np.multiply(Nj, np.sin(np.pi * x)))


m = 10
p = 2
k = 1
L = 2
neval = 3
spacing_func = lambda i: i * L / m
bc = (0.0, 1.0)

# Initialize problem and assemble matrices
brk = np.array([spacing_func(i) for i in range(0, m + 1)])
mesh = create_mesh(brk)
param_map = create_param_map(mesh)
space = create_fe_space(p, k, mesh)
ref_data = create_ref_data(neval, p, True)

A, b = assemble_fe_problem(mesh, space, ref_data, param_map, problem_B, problem_L, bc)

# Save system matrices
np.savetxt("1/A.txt", A, delimiter=",", fmt="%1.14f")
np.savetxt("1/b.txt", b, delimiter=",", fmt="%1.14f")

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

for l in range(mesh.elements.shape[1]):
    element = mesh.elements[:, l]
    xs, ns, dxns = eval_func(l, u_coefs, element, param_map, space, ref_data)
    xs_matrix[l, :] = xs
    ns_matrix[l, :] = ns
    dxns_matrix[l, :] = dxns

# Plot
Plotter.plot_results(
    xs_matrix, ns_matrix, dxns_matrix, "exercise_o.pdf", result_label="u_h"
)
