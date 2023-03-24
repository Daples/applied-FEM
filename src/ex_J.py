import numpy as np

from fem_students_1d import (
    create_mesh,
    create_param_map,
    create_fe_space,
    create_ref_data,
    assemble_fe_problem,
    problem_B,
    problem_L,
)
from utils.plotter import Plotter
from utils import eval_func
import matplotlib.pyplot as plt


m = 4
p = 2
k = 0
L = 2
neval = 3
spacing_func = lambda i: i**2 * L / m**2
bc = [0, 1]

brk = np.array([spacing_func(i) for i in range(0, m + 1)])
mesh = create_mesh(brk)
param_map = create_param_map(mesh)
space = create_fe_space(p, k, mesh)
ref_data = create_ref_data(neval, p, True)

A, b = assemble_fe_problem(mesh, space, ref_data, param_map, problem_B, problem_L, bc)
