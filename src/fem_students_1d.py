from typing import Callable

import numpy as np

from utils import eval_func
from fem.mesh import Mesh
from fem.param_map import ParamMap
from fem.reference_data import ReferenceData
from fem.space import Space


def assemble_fe_problem(
    mesh: Mesh,
    space: Space,
    ref_data: ReferenceData,
    param_map: ParamMap,
    problem_B: Callable,
    problem_L: Callable,
    bc: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    n = space.dim
    bar_A = np.zeros((n, n))
    bar_b = np.zeros(n)

    for l in range(mesh.elements.shape[1]):
        element = mesh.elements[:, l]

        xs = param_map.func(ref_data.evaluation_points, element[0], element[1])
        for i_index, i in enumerate(space.supported_bases[l, :]):
            ej_i = space.extraction_coefficients[l, i_index, :]
            ni = ej_i.dot(ref_data.reference_basis)
            dxni = param_map.imap_derivatives[l] * ej_i.dot(
                ref_data.reference_basis_derivatives
            )

            l_val = problem_L(xs, ni, dxni)
            val = np.sum(
                param_map.map_derivatives[l]
                * np.multiply(l_val, ref_data.quadrature_weights)
            )
            bar_b[i] += val

            for j_index, j in enumerate(space.supported_bases[l, :]):
                ej_i = space.extraction_coefficients[l, j_index, :]
                nj = ej_i.dot(ref_data.reference_basis)
                dxnj = param_map.imap_derivatives[l] * ej_i.dot(
                    ref_data.reference_basis_derivatives
                )
                b_val = problem_B(xs, ni, dxni, nj, dxnj)
                val = np.sum(
                    param_map.map_derivatives[l]
                    * np.multiply(b_val, ref_data.quadrature_weights)
                )
                bar_A[i, j] += val
    b = bar_b[1:-1] - bar_A[1:-1, 0] * bc[0] - bar_A[1:-1, -1] * bc[1]
    A = bar_A[1:-1, 1:-1]

    return A, b


def assemble_fe_mixed_problem(
    mesh: Mesh,
    spaces: list[Space],
    ref_datas: list[ReferenceData],
    param_maps: list[ParamMap],
    problem_B_mat: list[list[Callable]],
    problem_Ls: list[Callable],
) -> tuple[np.ndarray, np.ndarray]:
    ns = [space.dim for space in spaces]
    N = sum(ns)

    A = np.zeros((N, N))
    b = np.zeros(N)

    for l in range(mesh.elements.shape[1]):
        element = mesh.elements[:, l]

        accum = 0
        for space, param_map, ref_data, problem_Bs, problem_L in zip(
            spaces, param_maps, ref_datas, problem_B_mat, problem_Ls
        ):
            n = space.dim
            xs = param_map.func(ref_data.evaluation_points, element[0], element[1])
            for i_index, i in enumerate(space.supported_bases[l, :]):
                ej_i = space.extraction_coefficients[l, i_index, :]
                ni = ej_i.dot(ref_data.reference_basis)
                dxni = param_map.imap_derivatives[l] * ej_i.dot(
                    ref_data.reference_basis_derivatives
                )

                l_val = problem_L(xs, ni, dxni)
                val = np.sum(
                    param_map.map_derivatives[l]
                    * np.multiply(l_val, ref_data.quadrature_weights)
                )
                b[i + accum] += val

                accum_2 = 0
                for space_2, param_map_2, ref_data_2, problem_B in zip(
                    spaces, param_maps, ref_datas, problem_Bs
                ):
                    n_2 = space_2.dim
                    for j_index, j in enumerate(space_2.supported_bases[l, :]):
                        ej_i = space_2.extraction_coefficients[l, j_index, :]
                        nj = ej_i.dot(ref_data_2.reference_basis)
                        dxnj = param_map_2.imap_derivatives[l] * ej_i.dot(
                            ref_data_2.reference_basis_derivatives
                        )
                        b_val = problem_B(xs, ni, dxni, nj, dxnj)
                        val = np.sum(
                            param_map_2.map_derivatives[l]
                            * np.multiply(b_val, ref_data_2.quadrature_weights)
                        )
                        A[i + accum, j + accum_2] += val
                    accum_2 += n_2
            accum += n

    return A, b


def norm_0(
    func_u_e: Callable,
    u_h: np.ndarray,
    mesh: Mesh,
    param_map: ParamMap,
    space: Space,
    ref_data: ReferenceData,
) -> float:
    """"""

    norm = 0
    for l in range(mesh.elements.shape[1]):
        element = mesh.elements[:, l]
        xs, ns, _ = eval_func(l, u_h, element, param_map, space, ref_data)

        u_e = func_u_e(xs)
        func_eval = (ns - u_e) ** 2
        norm += np.sum(
            param_map.map_derivatives[l]
            * np.multiply(func_eval, ref_data.quadrature_weights)
        )

    return np.sqrt(norm)


def norm_1(
    func_u_e: Callable,
    d_func_u_e: Callable,
    u_h: np.ndarray,
    mesh: Mesh,
    param_map: ParamMap,
    space: Space,
    ref_data: ReferenceData,
) -> float:
    """"""

    norm = 0
    for l in range(mesh.elements.shape[1]):
        element = mesh.elements[:, l]
        xs, ns, dxns = eval_func(l, u_h, element, param_map, space, ref_data)

        u_e = func_u_e(xs)
        d_u_e = d_func_u_e(xs)
        func_eval = (ns - u_e) ** 2 + (dxns - d_u_e) ** 2
        norm += np.sum(
            param_map.map_derivatives[l]
            * np.multiply(func_eval, ref_data.quadrature_weights)
        )

    return np.sqrt(norm)
