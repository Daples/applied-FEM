from typing import Callable
import numpy as np

DataArray = list[float | int] | np.ndarray
FormData = np.ndarray | float
BilinearForm = Callable[[FormData, FormData, FormData, FormData, FormData], FormData]
LinearForm = Callable[[FormData, FormData, FormData], FormData]
