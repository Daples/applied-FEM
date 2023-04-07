import numpy as np

from fem_students_1d import (
    assemble_fe_problem,
    create_fe_space,
    create_mesh,
    create_param_map,
    create_ref_data,
)


def problem_B(
    _: np.ndarray, __: np.ndarray, dNj: np.ndarray, ___: np.ndarray, dNk: np.ndarray
) -> np.ndarray:
    return np.multiply(dNj, dNk)


def problem_L(_: np.ndarray, Nj: np.ndarray, __: np.ndarray) -> np.ndarray:
    return Nj


m = 4
p = 2
k = 0
L = 2
neval = 3
spacing_func = lambda i: i**2 * L / m**2
bc = (0.0, 1.0)

brk = np.array([spacing_func(i) for i in range(0, m + 1)])
mesh = create_mesh(brk)
param_map = create_param_map(mesh)
space = create_fe_space(p, k, mesh)
ref_data = create_ref_data(neval, p, True)

A, b = assemble_fe_problem(mesh, space, ref_data, param_map, problem_B, problem_L, bc)
