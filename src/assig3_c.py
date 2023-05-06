from typing import cast

import numpy as np

from fem.assembler import Assembler
from fem.mesh import Mesh
from fem.norms import h0_norm, h1_norm
from fem.param_map import ParametricMap
from fem.reference_data import ReferenceData
from fem.space import Space
from utils import eval_func
from utils.plotter import Plotter

problem_B = lambda x, Nj, dNj, Nk, dNk: np.multiply(dNj, dNk)
problem_L = (
    lambda x, Nj, dNj: (
        4
        * np.exp(-6 * np.power(x, 2))
        * (
            (-36 * x**2 + 4 * np.pi**2 + 3) * np.sin(4 * np.pi * x)
            + 24 * np.pi * x * np.cos(4 * np.pi * x)
        )
    )
    * Nj
)

u_e = lambda x: np.exp(-6 * x**2) * np.sin(4 * np.pi * x)
d_u_e = (
    lambda x: 4
    * np.exp(-6 * x**2)
    * (np.pi * np.cos(4 * np.pi * x) - 3 * x * np.sin(4 * np.pi * x))
)

p = 2
k = 1
L = 1
neval = 3
spacing_func = lambda i: i * L / m
bc = (0, 0)

ms = [4 * 2**i for i in range(0, 6)]
norms_0 = []
norms_1 = []
for m in ms:
    brk = np.array([spacing_func(i) for i in range(0, m + 1)])
    mesh = Mesh(brk)
    param_map = ParametricMap(mesh)
    space = Space(p, k, mesh)
    ref_data = ReferenceData(neval, p, True)

    A, b = Assembler.one_dimensional(
        mesh, space, ref_data, param_map, problem_B, problem_L, bc
    )

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
    x_exact = np.linspace(0, L, m * q)
    exact = u_e(x_exact)
    d_exact = d_u_e(x_exact)
    Plotter.plot_results(
        xs_matrix,
        ns_matrix,
        dxns_matrix,
        "ass_exercise_C.pdf",
        result_label="u_h",
        x_exact=x_exact,
        exact_solution=exact,
        d_exact_solution=d_exact,
    )

    # Save norms
    norms_0.append(h0_norm(u_e, u_coefs, mesh, param_map, space, ref_data))
    norms_1.append(h1_norm(u_e, d_u_e, u_coefs, mesh, param_map, space, ref_data))

conv_h0 = [1 / i**3 for i in ms]
conv_h1 = [1 / i**2 for i in ms]

const0 = [i / j for i, j in zip(norms_0, conv_h0)]
const1 = [i / j for i, j in zip(norms_1, conv_h1)]

const0 = cast(float, np.mean(const0))
const1 = cast(float, np.mean(const1))

Plotter.get_log_plot(
    ms,
    norms_0,
    "norm_0_C.pdf",
    lambda x: const0 * 1 / x**3,
    "$||e||_0$",
    "$C_0 \cdot h^3$",
)
Plotter.get_log_plot(
    ms,
    norms_1,
    "norm_1_C.pdf",
    lambda x: const1 * 1 / x**2,
    "$||e||_1$",
    "$C_1 \cdot h^2$",
)
