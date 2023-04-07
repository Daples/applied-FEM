from dataclasses import dataclass

import numpy as np


@dataclass
class Space:
    """A data class to represent the finite element space.

    Attributes
    ----------
    dim: int
        The dimension of the finite element space.
    supported_bases: numpy.ndarray
        The indices (columns) of the basis functions that are nonzero on the (row)
        element. Matrix of size `m x (p+1)`.
    extraction_coefficients: numpy.ndarray
        The coefficients of the reference basis functions on each element. Tensor of
        size `m x (p+1) x (p+1)`.
    """

    dim: int
    supported_bases: np.ndarray
    extraction_coefficients: np.ndarray
