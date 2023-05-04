from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceData:
    """A data class to store the reference data.

    Attributes
    ----------
    deg: int
        The polynomial degree of the finite element space basis functions.
    reference_element: numpy.ndarray
        The reference element ([0, 1] usually).
    evaluation_points: numpy.ndarray
        The evaluation points of the reference basis functions. Vector of size `q`.
    quadrature_weights: numpy.ndarray
        The quadrature weights at each evalaution point. Vector of size `q`.
    reference_basis: numpy.ndarray
        The evaluation of each reference basis function at the evaluation points. Matrix
        of size `(p+1) x deg`.
    reference_basis_derivatives: numpy.ndarray
        The derivatives of the matrix above. Matrix of size `(p+1) x deg`.
    """

    deg: int
    reference_element: np.ndarray
    evaluation_points: np.ndarray
    quadrature_weights: np.ndarray
    reference_basis: np.ndarray
    reference_basis_derivatives: np.ndarray
