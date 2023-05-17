import numpy as np

from fem.mesh import Mesh


class SupportExtractor:
    def __init__(
        self, supported_bases: np.ndarray, extraction_coefficients: np.ndarray
    ) -> None:
        self.supported_bases: np.ndarray = supported_bases
        self.extraction_coefficients: np.ndarray = extraction_coefficients


class Space2D:
    """A data class to represent the 2D finite element space.

    Attributes
    ----------
    """

    def __init__(
        self,
        n: int,
        boundary_bases: np.ndarray,
        support_extractors: list[SupportExtractor],
    ) -> None:
        self.n: int = n
        self.boundary_bases: np.ndarray = boundary_bases
        self.support_extractors: list[SupportExtractor] = support_extractors
