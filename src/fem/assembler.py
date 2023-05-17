from typing import Callable

import numpy as np

from fem.mesh import Mesh
from fem.param_map import ParametricMap
from fem.reference_data import ReferenceData
from fem.space import Space
from fem.create_geometric_map import create_geometric_map
from fem.create_ref_data import create_ref_data
from utils._typing import BilinearForm, LinearForm, BilinearForm_2D, LinearForm_2D


class Assembler:
    """A class to summarize the assembler of the FE problem."""

    @staticmethod
    def one_dimensional(
        mesh: Mesh,
        space: Space,
        ref_data: ReferenceData,
        param_map: ParametricMap,
        problem_B: BilinearForm,
        problem_L: LinearForm,
        bc: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assembles a one dimensional FE problem for the given data.

        Parameters
        ----------
        mesh: fem.mesh.Mesh
            The mesh on the problem domain.
        space: fem.space.Space
            The finite element space.
        ref_data: fem.ref_data.ReferenceData
            The reference element data.
        param_map: fem.param_map.ParametricMap
            The parametric map between the real and reference elements.
        problem_B: utils._typing.BilinearForm
            The bilinear form in the weak formulation.
        problem_L: utils._typing.LinearForm
            The linear form in the weak formulation.
        bc: tuple[float, float]
            The boundary conditions.

        Returns
        -------
        numpy.ndarray
            The left-hand side linear system matrix.
        numpy.ndarray
            The right-hand side linear system vector.
        """

        n = space.dim
        bar_A = np.zeros((n, n))
        bar_b = np.zeros(n)

        for l in range(mesh.elements.shape[1]):
            element = mesh.elements[:, l]

            xs = param_map.func(ref_data.evaluation_points, element[0], element[1])
            for i_index, i in enumerate(space.supported_bases[l, :]):
                ej_i = space.extraction_coefficients[l, i_index, :]
                ni = ej_i.dot(ref_data.reference_basis)
                dxni = param_map.imap_derivatives[l] * ej_i.dot(
                    ref_data.reference_basis_derivatives
                )

                l_val = problem_L(xs, ni, dxni)
                val = np.sum(
                    param_map.map_derivatives[l]
                    * np.multiply(l_val, ref_data.quadrature_weights)
                )
                bar_b[i] += val

                for j_index, j in enumerate(space.supported_bases[l, :]):
                    ej_i = space.extraction_coefficients[l, j_index, :]
                    nj = ej_i.dot(ref_data.reference_basis)
                    dxnj = param_map.imap_derivatives[l] * ej_i.dot(
                        ref_data.reference_basis_derivatives
                    )
                    b_val = problem_B(xs, ni, dxni, nj, dxnj)
                    val = np.sum(
                        param_map.map_derivatives[l]
                        * np.multiply(b_val, ref_data.quadrature_weights)
                    )
                    bar_A[i, j] += val
        b = bar_b[1:-1] - bar_A[1:-1, 0] * bc[0] - bar_A[1:-1, -1] * bc[1]
        A = bar_A[1:-1, 1:-1]

        return A, b

    @staticmethod
    def mixed_one_dimensional(
        mesh: Mesh,
        spaces: list[Space],
        ref_datas: list[ReferenceData],
        param_maps: list[ParametricMap],
        problem_B_mat: list[list[Callable]],
        problem_Ls: list[Callable],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assembles a system of one dimensional FE problems for the given data.

        Parameters
        ----------
        mesh: fem.mesh.Mesh
            The mesh on the problem domain.
        spaces: list[fem.space.Space]
            The list of finite element spaces.
        ref_datas: list[fem.ref_data.ReferenceData]
            The list of reference reference elements data.
        param_maps: list[fem.param_map.ParametricMap]
            The list of parametric maps between the real and reference elements.
        problem_B_mat: list[list[utils._typing.BilinearForm]]
            The matrix of bilinear forms in the weak formulation.
        problem_Ls: utils._typing.LinearForm
            The list of linear forms in the weak formulation.

        Returns
        -------
        numpy.ndarray
            The left-hand side linear system matrix.
        numpy.ndarray
            The right-hand side linear system vector.
        """

        ns = [space.dim for space in spaces]
        N = sum(ns)

        A = np.zeros((N, N))
        b = np.zeros(N)

        for l in range(mesh.elements.shape[1]):
            element = mesh.elements[:, l]

            accum = 0
            for space, param_map, ref_data, problem_Bs, problem_L in zip(
                spaces, param_maps, ref_datas, problem_B_mat, problem_Ls
            ):
                n = space.dim
                xs = param_map.func(ref_data.evaluation_points, element[0], element[1])
                for i_index, i in enumerate(space.supported_bases[l, :]):
                    ej_i = space.extraction_coefficients[l, i_index, :]
                    ni = ej_i.dot(ref_data.reference_basis)
                    dxni = param_map.imap_derivatives[l] * ej_i.dot(
                        ref_data.reference_basis_derivatives
                    )

                    l_val = problem_L(xs, ni, dxni)
                    val = np.sum(
                        param_map.map_derivatives[l]
                        * np.multiply(l_val, ref_data.quadrature_weights)
                    )
                    b[i + accum] += val

                    accum_2 = 0
                    for space_2, param_map_2, ref_data_2, problem_B in zip(
                        spaces, param_maps, ref_datas, problem_Bs
                    ):
                        n_2 = space_2.dim
                        for j_index, j in enumerate(space_2.supported_bases[l, :]):
                            ej_i = space_2.extraction_coefficients[l, j_index, :]
                            nj = ej_i.dot(ref_data_2.reference_basis)
                            dxnj = param_map_2.imap_derivatives[l] * ej_i.dot(
                                ref_data_2.reference_basis_derivatives
                            )
                            b_val = problem_B(xs, ni, dxni, nj, dxnj)
                            val = np.sum(
                                param_map_2.map_derivatives[l]
                                * np.multiply(b_val, ref_data_2.quadrature_weights)
                            )
                            A[i + accum, j + accum_2] += val
                        accum_2 += n_2
                accum += n

        return A, b

    @staticmethod
    def two_dimensional(
        fe_space,
        ref_data,
        geom_map,
        fe_geometry,
        problem_B: BilinearForm_2D,
        problem_L: LinearForm_2D
    ) -> tuple[np.ndarray, np.ndarray]:

        n = fe_space.n
        bar_A = np.zeros((n, n))
        bar_b = np.zeros(n)

        for l in range(fe_geometry.m):

            xs = geom_map.map[:,:,l]
            support_extractors = fe_space.support_extractors[l]
            det = np.multiply(
                    geom_map.map_derivatives[:,0,l], 
                    geom_map.map_derivatives[:,3,l]
                ) - np.multiply(
                    geom_map.map_derivatives[:,1,l], 
                    geom_map.map_derivatives[:,2,l]
                )
            for i_index, i in enumerate(support_extractors.supported_bases):
                ej_i = support_extractors.extraction_coefficients[i_index, :]
                ni = ej_i.dot(ref_data.reference_basis)
                dxni = (
                    ej_i.dot(np.multiply(geom_map.imap_derivatives[:,0,l], ref_data.reference_basis_derivatives[:,:,0])
                            +np.multiply(geom_map.imap_derivatives[:,1,l], ref_data.reference_basis_derivatives[:,:,1]))
                )
                dyni = (
                    ej_i.dot(np.multiply(geom_map.imap_derivatives[:,2,l], ref_data.reference_basis_derivatives[:,:,0])
                            +np.multiply(geom_map.imap_derivatives[:,3,l], ref_data.reference_basis_derivatives[:,:,1]))
                )

                l_val = problem_L(xs, ni, dxni, dyni)

                val = np.sum(
                    np.multiply(det,
                        np.multiply(l_val, ref_data.quadrature_weights)
                    ))
                bar_b[i] += val

                for j_index, j in enumerate(support_extractors.supported_bases):
                    ej_i = support_extractors.extraction_coefficients[j_index, :]
                    nj = ej_i.dot(ref_data.reference_basis)
                    dxnj = (
                        ej_i.dot(np.multiply(geom_map.imap_derivatives[:,0,l], ref_data.reference_basis_derivatives[:,:,0])
                                +np.multiply(geom_map.imap_derivatives[:,1,l], ref_data.reference_basis_derivatives[:,:,1]))
                    )
                    dynj = (
                        ej_i.dot(np.multiply(geom_map.imap_derivatives[:,2,l], ref_data.reference_basis_derivatives[:,:,0])
                                +np.multiply(geom_map.imap_derivatives[:,3,l], ref_data.reference_basis_derivatives[:,:,1]))
                    )

                    b_val = problem_B(xs, ni, dxni, dyni, nj, dxnj, dynj)

                    val = np.sum(
                        np.multiply(det,
                        np.multiply(b_val, ref_data.quadrature_weights)
                    ))

                    bar_A[i, j] += val
        
        idxs = set(fe_space.boundary_bases.squeeze().tolist())
        all_idxs = set(range(bar_A.shape[0]))
        idxs = all_idxs.difference(idxs)

        A = bar_A[idxs, idxs]
        b = bar_b[idxs]

        return A, b