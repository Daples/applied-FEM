from typing import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class ParamMap:
    func: Callable[
        [np.ndarray | float, np.ndarray | float, np.ndarray | float], np.ndarray | float
    ]
    map_derivatives: np.ndarray
    imap_derivatives: np.ndarray
