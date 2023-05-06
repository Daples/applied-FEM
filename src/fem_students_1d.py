from typing import Callable

import numpy as np

from fem.mesh import Mesh
from fem.param_map import ParametricMap
from fem.reference_data import ReferenceData
from fem.space import Space
from utils import eval_func


def norm_0(
    func_u_e: Callable,
    u_h: np.ndarray,
    mesh: Mesh,
    param_map: ParametricMap,
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
    param_map: ParametricMap,
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
