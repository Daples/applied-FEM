import numpy as np

from fem_students_1d import (
    assemble_fe_mixed_problem,
)
from utils import eval_func
from utils.plotter import Plotter
from fem.mesh import Mesh
from fem.param_map import ParamMap
from fem.reference_data import ReferenceData
from fem.space import Space

# Exact solutions
u_e = lambda x: np.cos(np.pi * (x - 0.5))
d_u_e = lambda x: -np.pi * np.sin(np.pi * (x - 0.5))
sigma = lambda x: -d_u_e(x)
d_sigma = lambda x: np.pi**2 * np.cos(np.pi * (x - 0.5))
fs = [sigma, u_e]
dfs = [d_sigma, d_u_e]


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
ps = [2, 1]
ks = [1, 0]
L = 1
neval = 3
spacing_func = lambda i: i * L / m

spaces = []
ref_datas = []
refined_ref_datas = []
param_maps = []
brk = np.array([spacing_func(i) for i in range(0, m + 1)])
mesh = Mesh(brk)
param_map = ParamMap(mesh)

for i, p in enumerate(ps):
    k = ks[i]
    space = Space(p, k, mesh)
    ref_data = ReferenceData(neval, p, True)
    spaces.append(space)
    ref_datas.append(ref_data)
    param_maps.append(param_map)

    # Refined
    neval_refined = 20
    refined_ref_datas.append(ReferenceData(neval_refined, p, True))

A, b = assemble_fe_mixed_problem(
    mesh, spaces, ref_datas, param_maps, problem_B_mat, problem_Ls
)
x_sol = np.linalg.solve(A, b)

ns = [space.dim for space in spaces]
accum = 0
labels = ["\\sigma", "u"]
for i, n in enumerate(ns):
    space = spaces[i]
    ref_data = refined_ref_datas[i]
    param_map = param_maps[i]
    coefs = x_sol[accum : accum + n]
    accum += n

    # Recover solution
    m = mesh.elements.shape[1]
    q = ref_data.evaluation_points.shape[0]
    xs_matrix = np.zeros((m, q))
    ns_matrix = np.zeros_like(xs_matrix)
    dxns_matrix = np.zeros_like(ns_matrix)

    for current_element in range(mesh.elements.shape[1]):
        element = mesh.elements[:, current_element]
        xs, ns, dxns = eval_func(
            current_element, coefs, element, param_map, space, ref_data
        )
        xs_matrix[current_element, :] = xs
        ns_matrix[current_element, :] = ns
        dxns_matrix[current_element, :] = dxns

    # Plot
    x_exact = np.linspace(0, L, m * q)
    exact = fs[i](x_exact)
    d_exact = dfs[i](x_exact)
    name = labels[i].replace("\\", "")
    Plotter.plot_results(
        xs_matrix,
        ns_matrix,
        dxns_matrix,
        f"ass_exercise_G_{labels[i]}.pdf",
        result_label=f"{labels[i]}_h",
        x_exact=x_exact,
        exact_solution=exact,
        d_exact_solution=d_exact,
    )
