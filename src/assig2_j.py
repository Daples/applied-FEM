import numpy as np

from fem.mesh import Mesh
from fem.param_map import ParametricMap
from fem.reference_data import ReferenceData
from fem.space import Space
from fem.assembler import Assembler


problem_B = lambda x, Nj, dNj, Nk, dNk: np.multiply(dNj, dNk)
problem_L = lambda x, Nj, dNj: Nj

m = 4
p = 2
k = 0
L = 2
neval = 3
spacing_func = lambda i: i**2 * L / m**2
bc = (0.0, 1.0)

brk = np.array([spacing_func(i) for i in range(0, m + 1)])
mesh = Mesh(brk)
param_map = ParametricMap(mesh)
space = Space(p, k, mesh)
ref_data = ReferenceData(neval, p, True)

A, b = Assembler.one_dimensional(
    mesh, space, ref_data, param_map, problem_B, problem_L, bc
)
