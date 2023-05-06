from typing import Callable

import numpy as np

from fem.mesh import Mesh
from fem.param_map import ParametricMap
from fem.reference_data import ReferenceData
from fem.space import Space
from utils import eval_func
from utils._typing import FormData


def h0_norm(
    exact_sol: Callable[[FormData], FormData],
    u_h: np.ndarray,
    mesh: Mesh,
    param_map: ParametricMap,
    space: Space,
    ref_data: ReferenceData,
) -> float:
    """It approximates the H^0 norm between the exact and the approximated solution.

    Parameters
    ----------
    exact_sol: utils._typing.FormData -> utils._typing.FormData
        The exact solution callable.
    u_h: numpy.ndarray
        The solution coefficients from FE approximation.
    mesh: fem.mesh.Mesh
        The mesh on the problem domain.
    param_map: fem.param_map.ParametricMap
        The parametric map between the real and reference elements.
    space: fem.space.Space
        The finite element space.
    ref_data: fem.ref_data.ReferenceData
        The reference element data.

    Returns
    -------
    float
        The estimated norm.
    """

    norm = 0
    for l in range(mesh.elements.shape[1]):
        element = mesh.elements[:, l]
        xs, ns, _ = eval_func(l, u_h, element, param_map, space, ref_data)

        u_e = exact_sol(xs)
        func_eval = (ns - u_e) ** 2
        norm += np.sum(
            param_map.map_derivatives[l]
            * np.multiply(func_eval, ref_data.quadrature_weights)
        )

    return np.sqrt(norm)


def h1_norm(
    exact_sol: Callable[[FormData], FormData],
    d_exact_sol: Callable[[FormData], FormData],
    u_h: np.ndarray,
    mesh: Mesh,
    param_map: ParametricMap,
    space: Space,
    ref_data: ReferenceData,
) -> float:
    """It approximates the H^1 norm between the exact and the approximated solution.

    Parameters
    ----------
    exact_sol: utils._typing.FormData -> utils._typing.FormData
        The exact solution callable.
    d_exact_sol: utils._typing.FormData -> utils._typing.FormData
        The derivative of the exact solution callable.
    u_h: numpy.ndarray
        The solution coefficients from FE approximation.
    mesh: fem.mesh.Mesh
        The mesh on the problem domain.
    param_map: fem.param_map.ParametricMap
        The parametric map between the real and reference elements.
    space: fem.space.Space
        The finite element space.
    ref_data: fem.ref_data.ReferenceData
        The reference element data.

    Returns
    -------
    float
        The estimated norm.
    """

    norm = 0
    for l in range(mesh.elements.shape[1]):
        element = mesh.elements[:, l]
        xs, ns, dxns = eval_func(l, u_h, element, param_map, space, ref_data)

        u_e = exact_sol(xs)
        d_u_e = d_exact_sol(xs)
        func_eval = (ns - u_e) ** 2 + (dxns - d_u_e) ** 2
        norm += np.sum(
            param_map.map_derivatives[l]
            * np.multiply(func_eval, ref_data.quadrature_weights)
        )

    return np.sqrt(norm)
