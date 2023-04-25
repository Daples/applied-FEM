import numpy as np

from fem_students_1d import (
    assemble_fe_mixed_problem,
    create_fe_space,
    create_mesh,
    create_param_map,
    create_ref_data,
    norm_0,
    norm_1,
)
from utils import eval_func
from utils.plotter import Plotter


def problem_B11(
    x: np.ndarray, Nj: np.ndarray, _: np.ndarray, Nk: np.ndarray, dNk: np.ndarray
) -> np.ndarray:
    return np.multiply(Nj, Nk)


def problem_B12(
    x: np.ndarray, Nj: np.ndarray, dNj: np.ndarray, Nk: np.ndarray, dNk: np.ndarray
) -> np.ndarray:
    return -np.multiply(dNj, Nk)


def problem_B21(
    x: np.ndarray, Nj: np.ndarray, dNj: np.ndarray, Nk: np.ndarray, dNk: np.ndarray
) -> np.ndarray:
    return np.multiply(Nj, dNk)


def problem_B22(
    x: np.ndarray, Nj: np.ndarray, dNj: np.ndarray, Nk: np.ndarray, dNk: np.ndarray
) -> np.ndarray:
    return np.zeros_like(Nj)


def problem_L1(x: np.ndarray, Nj: np.ndarray, __: np.ndarray) -> np.ndarray:
    return np.zeros_like(Nj)


def problem_L2(x: np.ndarray, Nj: np.ndarray, Nk: np.ndarray) -> np.ndarray:
    return np.pi**2 * np.multiply(np.cos(np.pi * (x - 0.5)), Nj)


problem_B_mat = [[problem_B11, problem_B12], [problem_B21, problem_B22]]
problem_Ls = [problem_L1, problem_L2]

# Choice 1
m = 14
ps = [1, 1]
ks = [0, 0]
L = 1
neval = 3
spacing_func = lambda i: i * L / m

spaces = []
ref_datas = []
param_maps = []
brk = np.array([spacing_func(i) for i in range(0, m + 1)])
mesh = create_mesh(brk)
for i, p in enumerate(ps):
    k = ks[i]
    param_map = create_param_map(mesh)
    space = create_fe_space(p, k, mesh)
    ref_data = create_ref_data(neval, p, True)
    spaces.append(space)
    ref_datas.append(ref_data)
    param_maps.append(param_map)

A, b = assemble_fe_mixed_problem(
    mesh, spaces, ref_datas, param_maps, problem_B_mat, problem_Ls
)
x_sol = np.linalg.solve(A, b)
