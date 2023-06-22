import numpy as np
from fem.fe_geometry import FEGeometry
from fem.bi_dim_reference_data import BidimensionalReferenceData


class BidimensionalGeometricMap:
    def __init__(
        self, fe_geometry: FEGeometry, ref_data: BidimensionalReferenceData
    ) -> None:
        self.map: np.ndarray = np.zeros(0)
        self.map_derivatives: np.ndarray = np.zeros(0)
        self.imap_derivatives: np.ndarray = np.zeros(0)
        self.__init_ref_data__(fe_geometry, ref_data)

    def __init_ref_data__(
        self, fe_geometry: FEGeometry, ref_data: BidimensionalReferenceData
    ) -> None:
        """It initializes the reference data with the given inputs.

        Parameters
        ----------
        fe_geometry: fem.fe_geometry.FEGeometry
            The finite element geometry object.
        ref_data: fem.bi_dim_reference_data.BidimensionalReferenceData
            The bidimensional reference data object.
        """

        nqs = ref_data.neval**2
        self.map = np.zeros((nqs, 2, fe_geometry.m))
        self.map_derivatives = np.zeros((nqs, 4, fe_geometry.m))
        self.imap_derivatives = np.zeros((nqs, 4, fe_geometry.m))

        for i in range(fe_geometry.m):
            for j in range(2):
                self.map[:, j, i] = np.matmul(
                    fe_geometry.map_coefficients[:, j, i], ref_data.reference_basis
                )
                for k in range(2):
                    self.map_derivatives[:, k * 2 + j, i] = np.matmul(
                        fe_geometry.map_coefficients[:, j, i],
                        ref_data.reference_basis_derivatives[:, :, k],
                    )

        det = np.multiply(
            self.map_derivatives[:, 0, :], self.map_derivatives[:, 3, :]
        ) - np.multiply(self.map_derivatives[:, 1, :], self.map_derivatives[:, 2, :])

        aux = [3, 1, 2, 0]
        aux2 = [1, -1, -1, 1]

        for i in range(4):
            self.imap_derivatives[:, i, :] = (
                (1 / det) * aux2[i] * self.map_derivatives[:, aux[i], :]
            )
