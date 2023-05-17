from reference_data import ReferenceData
import numpy as np


class create_ref_data:

    neval: int
    deg: list
    evaluation_points: np.ndarray
    quadrature_weights: np.ndarray
    reference_basis: np.ndarray
    reference_basis_derivatives: np.ndarray

    def __init__(self, neval: int, deg: list, integrate: bool = False) -> None:
        self.neval: int = neval
        self.deg: list = deg
        self.evaluation_points: np.ndarray = np.zeros(0)
        self.quadrature_weights: np.ndarray = np.zeros(0)
        self.reference_basis: np.ndarray = np.zeros(0)
        self.reference_basis_derivatives: np.ndarray = np.zeros(0)
        self.__init_ref_data__(neval, deg, integrate)

    def __init_ref_data__(self, neval: int, deg: list, integrate: bool) -> None:
        ref_data_x = ReferenceData(neval, deg[0], True)
        ref_data_y = ReferenceData(neval, deg[1], True)

        n_fs = (deg[0] + 1) * (deg[1] + 1)
        n_qs = neval**2

        self.evaluation_points = np.zeros((n_qs, 2))
        self.quadrature_weights = np.zeros((n_qs))

        self.reference_basis = np.zeros((n_fs, n_qs))
        self.reference_basis_derivatives = np.zeros((n_fs, n_qs, 2))

        for i in range(neval):
            for j in range(neval):
                p_index = j * neval + i
                self.evaluation_points[p_index, :] = np.array(
                    ([ref_data_x.evaluation_points[j], ref_data_y.evaluation_points[i]])
                )
                self.quadrature_weights[p_index] = (
                    ref_data_x.quadrature_weights[j] * ref_data_y.quadrature_weights[i]
                )

                for k in range(deg[0] + 1):
                    for l in range(deg[1] + 1):
                        f_index = k + (deg[0] + 1) * l
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
