from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    """A data class to represent a mesh.

    Attributes
    ----------
    m: int
        The number of elements in the mesh.
    elements: numpy.ndarray
        The elements matrix, where every column rerpesent an element.
    """

    m: int
    elements: np.ndarray
