import numpy as np


class Mesh:
    """A data class to represent a mesh.

    Attributes
    ----------
    m: int
        The number of elements in the mesh.
    elements: numpy.ndarray
        The elements matrix, where every column rerpesent an element.
    """

    def __init__(self, partition_points: np.ndarray) -> None:
        self.m: int = 0
        self.elements: np.ndarray = np.zeros(0)
        self.__init_mesh__(partition_points)

    def __init_mesh__(self, brk: np.ndarray) -> None:
        """It initializes the array of elements.

        Parameters
        ----------
        brk: numpy.ndarray
            The interval partitions.
        """

        self.m = brk.shape[0] - 1
        elements = np.zeros((2, self.m))
        elements[0, :] = brk[:-1]
        elements[1, :] = brk[1:]
        self.elements = brk
