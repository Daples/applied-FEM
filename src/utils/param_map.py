from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ParamMap:
    """A data class to represent a parametric map.

    Attributes
    ----------
    func: [
        numpy.ndarray | float, numpy.ndarray | float, numpy.ndarray | float
    ] -> numpy.ndarray | float
        The map from the reference element to the actual element.

    map_derivatives: numpy.ndarray
        The derivatives of the map on the reference element.
    imap_derivatives: numpy.ndarray
        The derivatives of the inverse map.
    """

    func: Callable[
        [np.ndarray | float, np.ndarray | float, np.ndarray | float], np.ndarray | float
    ]
    map_derivatives: np.ndarray
    imap_derivatives: np.ndarray
