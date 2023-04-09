from typing import cast

import numpy as np

from utils.param_map import ParamMap
from utils.reference_data import ReferenceData
from utils.space import Space


def eval_func(
    current_element: int,
    coefs: np.ndarray,
    element: np.ndarray,
    param_map: ParamMap,
    space: Space,
    ref_data: ReferenceData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """It evaluates a function based on coefficients, basis functions and function
    domain.

    Parameters
    ----------
    curent_element: int
        The index of the current element.
    coefs: numpy.ndarray
        The coefficients of the basis functions for the current element.
    element: numpy.ndarray
        The current element.
    param_map: utils.param_map.ParamMap
        The parametric map object.
    space: utils.space.Space
        The finite element space object.
    ref_data: utils.reference_data.ReferenceData
        The reference element object representation.
    """

    xs = cast(
        np.ndarray, param_map.func(ref_data.evaluation_points, element[0], element[1])
    )
    ns = np.zeros_like(xs)
    dxns = np.zeros_like(xs)
    for i, j in enumerate(space.supported_bases[current_element, :]):
        ej_i = space.extraction_coefficients[current_element, i, :]
        ns += coefs[j] * ej_i.dot(ref_data.reference_basis)
        dxns += (
            coefs[j]
            * param_map.imap_derivatives[current_element]
            * ej_i.dot(ref_data.reference_basis_derivatives)
        )

    return xs, ns, dxns
