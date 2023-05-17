import numpy as np


class FEGeometry:
    """"""

    def __init__(self, m: int, map_coefficients: np.ndarray) -> None:
        self.m: int = m
        self.map_coefficients: np.ndarray = map_coefficients
