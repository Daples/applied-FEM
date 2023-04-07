from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss as gaussquad
from scipy.interpolate import _bspl as bspl

from utils.mesh import Mesh
from utils.param_map import ParamMap
from utils.reference_data import ReferenceData
from utils.space import Space


def create_ref_data(neval: int, deg: int, integrate: bool = False) -> ReferenceData:
    # reference unit domain
    reference_element = np.array([0, 1])
    if integrate is False:
        # point for plotting are equispaced on reference element
        x = np.linspace(reference_element[0], reference_element[1], neval)
        evaluation_points = x
        quadrature_weights = np.zeros((0,))
    else:
        # points (and weights) for integration are computed according to Gauss quadrature
        x, w = gaussquad(neval)
        evaluation_points = 0.5 * (x + 1)
        quadrature_weights = w / 2
    # knots for defining B-splines
    knt = np.concatenate(
        (np.zeros((deg + 1,), dtype=float), np.ones((deg + 1,), dtype=float)), axis=0
    )
    # reference basis function values
    tmp = [
        bspl.evaluate_all_bspl(knt, deg, evaluation_points[i], deg, nu=0)  # type: ignore
        for i in range(evaluation_points.shape[0])
    ]
    reference_basis = np.vstack(tmp).T
    # reference basis function first derivatives
    tmp = [
        bspl.evaluate_all_bspl(knt, deg, evaluation_points[i], deg, nu=1)  # type: ignore
        for i in range(evaluation_points.shape[0])
    ]
    reference_basis_derivatives = np.vstack(tmp).T

    return ReferenceData(
        deg,
        reference_element,
        evaluation_points,
        quadrature_weights,
        reference_basis,
        reference_basis_derivatives,
    )


def create_fe_space(deg, reg, mesh) -> Space:
    def bezier_extraction(knt: np.ndarray, deg: int) -> np.ndarray:
        # breakpoints
        brk = np.unique(knt)
        # number of elements
        nel = brk.shape[0] - 1
        # number of knots
        m = knt.shape[0]
        # assuming an open knotvector, knt[a] is the last repetition of the first knot
        a = deg
        # next knot
        b = a + 1
        # Bezier element being processed
        nb = 0
        # first extraction matrix
        C = [np.eye(deg + 1, deg + 1, dtype=float)]
        # this is where knot-insertion coefficients are saved
        alphas = np.zeros((deg - 1,), dtype=float)
        while b < m:
            # initialize extraction matrix for next element
            C.append(np.eye(deg + 1, deg + 1))
            # save index of current knot
            i = b
            # find last occurence of current knot
            while b < m - 1 and knt[b + 1] == knt[b]:
                b += 1
            # multiplicity of current knot
            mult = b - i + 1
            # if multiplicity is < deg, smoothness is at least C0 and extraction may differ from an identity matrix
            if mult < deg:
                numer = knt[b] - knt[a]
                # smoothness of splines
                r = deg - mult
                # compute linear combination coefficients
                for j in range(deg - 1, mult - 1, -1):
                    alphas[j - mult] = numer / (knt[a + j + 1] - knt[a])
                for j in range(r):
                    s = mult + j
                    for k in range(deg, s, -1):
                        alpha = alphas[k - s - 1]
                        C[nb][:, k] = (
                            alpha * C[nb][:, k] + (1.0 - alpha) * C[nb][:, k - 1]
                        )
                    save = r - j
                    if b < m:
                        C[nb + 1][save - 1 : j + save + 1, save - 1] = C[nb][
                            deg - j - 1 : deg + 1, deg
                        ]
            # increment element index
            nb += 1
            if b < m:
                a = b
                b += 1
            C = C[:nel]
        return np.array(C)

    # number of mesh elements
    nel = mesh.m
    # unique breakpoints
    if nel == 1:
        brk = mesh.elements.T[0]
    else:
        brk = np.concatenate(
            (mesh.elements[0], np.array([mesh.elements[1][-1]])), axis=0
        )
    # multiplicity of each breakpoint
    mult = deg - reg
    # knot vector for B-spline definition
    knt = np.concatenate(
        (
            np.ones((deg + 1,), dtype=float) * brk[0],
            np.ones((deg + 1,), dtype=float) * brk[-1],
            np.repeat(brk[1:-1], mult),
        ),
        axis=0,
    )
    knt = np.sort(knt)
    # coefficients of linear combination
    C = bezier_extraction(knt, deg)
    # dimension of finite element space
    dim = knt.shape[0] - deg - 1
    # connectivity information (i.e., which bases are non-zero on which element)
    econn = np.zeros((nel, deg + 1), dtype=int)
    for i in range(nel):
        if i == 0:
            econn[i] = np.arange(deg + 1)
        else:
            econn[i] = econn[i - 1] + mult

    return Space(dim, econn, C)


def create_mesh(brk: np.ndarray) -> Mesh:
    m = brk.shape[0] - 1
    elements = np.zeros((2, m))
    elements[0, :] = brk[:-1]
    elements[1, :] = brk[1:]

    return Mesh(m, elements)


def create_param_map(mesh: Mesh) -> ParamMap:
    def func(
        c: float | np.ndarray, lower: np.ndarray | float, upper: np.ndarray | float
    ) -> np.ndarray | float:
        return lower + c * (upper - lower)

    map_derivatives = mesh.elements[1, :] - mesh.elements[0, :]  # type: ignore
    imap_derivatives = 1 / map_derivatives
    return ParamMap(func, map_derivatives, imap_derivatives)


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
