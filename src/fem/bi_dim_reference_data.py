from fem.reference_data import ReferenceData
import numpy as np


class BidimensionalReferenceData:
    def __init__(self, neval: int, deg: list, integrate: bool = False) -> None:
        self.neval: int = neval
        self.deg: list[int] = deg
        self.evaluation_points: np.ndarray = np.zeros(0)
        self.quadrature_weights: np.ndarray = np.zeros(0)
        self.reference_basis: np.ndarray = np.zeros(0)
        self.reference_basis_derivatives: np.ndarray = np.zeros(0)
        self.__init_ref_data__()

    def __init_ref_data__(self) -> None:
        ref_data_x = ReferenceData(self.neval, self.deg[0], True)
        ref_data_y = ReferenceData(self.neval, self.deg[1], True)

        n_fs = (self.deg[0] + 1) * (self.deg[1] + 1)
        n_qs = self.neval**2

        self.evaluation_points = np.zeros((n_qs, 2))
        self.quadrature_weights = np.zeros((n_qs))

        self.reference_basis = np.zeros((n_fs, n_qs))
        self.reference_basis_derivatives = np.zeros((n_fs, n_qs, 2))

        for i in range(self.neval):
            for j in range(self.neval):
                p_index = j * self.neval + i
                self.evaluation_points[p_index, :] = np.array(
                    ([ref_data_x.evaluation_points[j], ref_data_y.evaluation_points[i]])
                )
                self.quadrature_weights[p_index] = (
                    ref_data_x.quadrature_weights[j] * ref_data_y.quadrature_weights[i]
                )

                for k in range(self.deg[0] + 1):
                    for l in range(self.deg[1] + 1):
                        f_index = k + (self.deg[0] + 1) * l
                        self.reference_basis[f_index, p_index] = (
                            ref_data_x.reference_basis[k, i]
                            * ref_data_y.reference_basis[l, j]
                        )
                        self.reference_basis_derivatives[
                            f_index, p_index, :
                        ] = np.array(
                            (
                                [
                                    ref_data_x.reference_basis_derivatives[k, i]
                                    * ref_data_y.reference_basis[l, j],
                                    ref_data_x.reference_basis[k, i]
                                    * ref_data_y.reference_basis_derivatives[l, j],
                                ]
                            )
                        )
