from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    m: int
    elements: np.ndarray
