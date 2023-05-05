from typing import Callable

import numpy as np

from fem.mesh import Mesh


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

    def __init__(self, mesh: Mesh) -> None:
        self.func: Callable[
            [np.ndarray | float, np.ndarray | float, np.ndarray | float],
            np.ndarray | float,
        ] = lambda _, __, ___: 0
        self.map_derivatives: np.ndarray = np.zeros(0)
        self.imap_derivatives: np.ndarray = np.zeros(0)
        self.__init_param_map__(mesh)

    def __init_param_map__(self, mesh: Mesh) -> None:
        """Initialize the parametric map attributes.

        Parameters
        ----------
        mesh: fem.mesh.Mesh
            The problem mesh.
        """

        def func(
            c: float | np.ndarray, lower: np.ndarray | float, upper: np.ndarray | float
        ) -> np.ndarray | float:
            return lower + c * (upper - lower)

        self.func = func
        self.map_derivatives = mesh.elements[1, :] - mesh.elements[0, :]  # type: ignore
        self.imap_derivatives = 1 / self.map_derivatives
