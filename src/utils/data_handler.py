from typing import Callable
from utils._typing import DataArray

import numpy as np


class DataHandler:
    """A class to homogenize the management and typing of data."""

    array_type: Callable[[DataArray], np.ndarray]

    @classmethod
    def __cast_array__(cls, *args: DataArray) -> tuple[np.ndarray, ...]:
        """It converts the input array types into the standardize  type.

        Parameters
        ----------
        args: utils._typing.DataArray
            The arrays to convert.

        Returns
        -------
        tuple[utils._typing.DataArrray]
            The transformed types.
        """

        return tuple(map(lambda x: np.array(x), args))
