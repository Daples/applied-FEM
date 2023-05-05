import numpy as np

from fem.mesh import Mesh
from fem.param_map import ParamMap
from fem.reference_data import ReferenceData
from fem.space import Space
from fem_students_1d import assemble_fe_problem


def problem_B(
    _: np.ndarray, __: np.ndarray, dNj: np.ndarray, ___: np.ndarray, dNk: np.ndarray
) -> np.ndarray:
    return np.multiply(dNj, dNk)


def problem_L(_: np.ndarray, Nj: np.ndarray, __: np.ndarray) -> np.ndarray:
    return Nj


m = 4
p = 1
k = 0
L = 2
neval = 3
spacing_func = lambda i: i * L / m
bc = (0.0, 1.0)

brk = np.array([spacing_func(i) for i in range(0, m + 1)])
mesh = Mesh(brk)
param_map = ParamMap(mesh)
space = Space(p, k, mesh)
ref_data = ReferenceData(neval, p, True)

A, b = assemble_fe_problem(mesh, space, ref_data, param_map, problem_B, problem_L, bc)
