from typing import cast, Any

import numpy as np

from scipy.io import loadmat

from fem.param_map import ParametricMap
from fem.reference_data import ReferenceData
from fem.space import Space
from fem.space2D import SupportExtractor, Space2D
from fem.fe_geometry import FEGeometry


def eval_func(
    current_element: int,
    coefs: np.ndarray,
    element: np.ndarray,
    param_map: ParametricMap,
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
    param_map: utils.param_map.ParametricMap
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


def read_mat(filename: str) -> tuple[Any, Any]:
    """"""

    mat = loadmat(filename)
    geometry_contents = mat["fe_geometry"][0][0]
    m = geometry_contents["m"][0, 0]
    map_coefficients = geometry_contents["map_coefficients"]
    fe_geometry = FEGeometry(m, map_coefficients)

    space_contents = mat["fe_space"][0][0]
    n = space_contents["n"][0, 0]
    boundary_bases = space_contents["boundary_bases"]
    support_and_extraction = space_contents["support_and_extraction"]

    support_extractors = []
    for elem in support_and_extraction:
        supported_bases = elem["supported_bases"]
        extration_coefficients = elem["extraction_coefficients"][0]
        support_extractors.append(
            SupportExtractor(supported_bases, extration_coefficients)
        )

    fe_space = Space2D(n, boundary_bases, support_extractors)
    return fe_geometry, fe_space
