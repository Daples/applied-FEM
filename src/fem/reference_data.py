import numpy as np

from numpy.polynomial.legendre import leggauss as gaussquad
from scipy.interpolate import _bspl as bspl


class ReferenceData:
    """A data class to store the reference data.

    Attributes
    ----------
    deg: int
        The polynomial degree of the finite element space basis functions.
    reference_element: numpy.ndarray
        The reference element ([0, 1] usually).
    evaluation_points: numpy.ndarray
        The evaluation points of the reference basis functions. Vector of size `q`.
    quadrature_weights: numpy.ndarray
        The quadrature weights at each evalaution point. Vector of size `q`.
    reference_basis: numpy.ndarray
        The evaluation of each reference basis function at the evaluation points. Matrix
        of size `(p+1) x deg`.
    reference_basis_derivatives: numpy.ndarray
        The derivatives of the matrix above. Matrix of size `(p+1) x deg`.
    """

    deg: int
    reference_element: np.ndarray
    evaluation_points: np.ndarray
    quadrature_weights: np.ndarray
    reference_basis: np.ndarray
    reference_basis_derivatives: np.ndarray

    def __init__(self, neval: int, deg: int, integrate: bool = False) -> None:
        self.deg: int = deg
        self.reference_element: np.ndarray = np.zeros(0)
        self.evaluation_points: np.ndarray = np.zeros(0)
        self.quadrature_weights: np.ndarray = np.zeros(0)
        self.reference_basis: np.ndarray = np.zeros(0)
        self.reference_basis_derivatives: np.ndarray = np.zeros(0)
        self.__init_ref_data__(neval, deg, integrate)

    def __init_ref_data__(self, neval: int, deg: int, integrate: bool) -> None:
        """It initializes the reference data. (Implemented by Deepesh Toshniwal)

        Parameters
        ----------
        neval: int
            The number of evaluation points on the reference element.
        deg: int
            The degree of the polynomial for the FE space.
        integrate: bool
            Flag to signal when quadrature coefficients are also required.
        """

        # Reference unit domain
        self.reference_element = np.array([0, 1])
        if integrate is False:
            # Equispaced points on reference element
            x = np.linspace(self.reference_element[0], self.reference_element[1], neval)
            self.evaluation_points = x
            self.quadrature_weights = np.zeros(0)
        else:
            # Gauss quadrature for integration
            x, w = gaussquad(neval)
            self.evaluation_points = 0.5 * (x + 1)
            self.quadrature_weights = w / 2

        # B-splines knots
        knt = np.concatenate(
            (np.zeros((deg + 1,), dtype=float), np.ones((deg + 1,), dtype=float)),
            axis=0,
        )
        # Reference basis function
        tmp = [
            bspl.evaluate_all_bspl(knt, deg, self.evaluation_points[i], deg, nu=0)  # type: ignore
            for i in range(self.evaluation_points.shape[0])
        ]
        self.reference_basis = np.vstack(tmp).T

        # Reference basis function first derivative
        tmp = [
            bspl.evaluate_all_bspl(knt, deg, self.evaluation_points[i], deg, nu=1)  # type: ignore
            for i in range(self.evaluation_points.shape[0])
        ]
        self.reference_basis_derivatives = np.vstack(tmp).T
