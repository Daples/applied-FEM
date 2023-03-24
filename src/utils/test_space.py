from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from fem_students_1d import (
    create_mesh,
    create_param_map,
    create_fe_space,
    create_ref_data,
)
from utils.plotter import Plotter


class Tests:
    @staticmethod
    def test_space(
        p: int,
        k: int,
        m: int,
        spacing_func: Callable[[int], float],
        neval: int,
        filename: str,
    ) -> None:
        """"""

        Plotter.__clear__()
        Plotter.__setup_config__()

        brk = np.array([spacing_func(i) for i in range(0, m + 1)])
        mesh = create_mesh(brk)
        param_map = create_param_map(mesh)
        space = create_fe_space(p, k, mesh)
        ref_data = create_ref_data(neval, p, False)

        n = space["n"]
        supported_bases = space["supported_bases"]
        extraction_coefficients = space["extraction_coefficients"]
        reference_basis = ref_data["reference_basis"]
        reference_basis_derivatives = ref_data["reference_basis_derivatives"]
        evaluation_points = ref_data["evaluation_points"]

        test_elements = [0, 3, n - 1]
        fig, axs = plt.subplots(len(test_elements), 2)
        fig.tight_layout()
        for h, j in enumerate(test_elements):
            for l in range(mesh.elements.shape[1]):
                element = mesh.elements[:, l]
                xs = param_map.func(evaluation_points, element[0], element[1])
                ns = np.zeros_like(xs)
                dxns = np.zeros_like(xs)

                aux = supported_bases[l] == j
                is_supported = aux.any()
                if is_supported:
                    index_support = np.where(aux)[0][0]
                    ej_i = extraction_coefficients[l][index_support, :]
                    ns = ej_i.dot(reference_basis)
                    dxns = param_map.imap_derivatives[l] * ej_i.dot(
                        reference_basis_derivatives
                    )
                axs[h][0].plot(xs, ns, "k")
                axs[h][1].plot(xs, dxns, "k")

            axs[h][0].set_xlabel("$x$")
            axs[h][0].set_ylabel("$N(x)$")
            axs[h][1].set_xlabel("$x$")
            axs[h][1].set_ylabel("$N'(x)$")
        plt.savefig(Plotter.add_folder(filename), bbox_inches="tight")

    @staticmethod
    def test_solution(coefs: np.ndarray):
        """"""
