from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from fem_students_1d import (
    create_fe_space,
    create_mesh,
    create_param_map,
    create_ref_data,
)
from utils import eval_func
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
        """It creates the plot of the first, fourth and last elements for testing.

        Parameters
        ----------
        p: int
            The degree of the polynomial.
        k: int
            The smoothness degree.
        m: int
            The number of evaluation points for each basis function.
        spacing_func: int -> float
            The function to map the partitions of the testing interval.
        neval: int
            The number of quadrature points.
        filename:
            The name of the output plot.
        """

        Plotter.__clear__()
        Plotter.__setup_config__()

        brk = np.array([spacing_func(i) for i in range(0, m + 1)])
        mesh = create_mesh(brk)
        param_map = create_param_map(mesh)
        space = create_fe_space(p, k, mesh)
        ref_data = create_ref_data(neval, p, False)

        n = space.dim

        test_elements = [0, 3, n - 1]
        fig, axs = plt.subplots(len(test_elements), 2)
        fig.tight_layout(pad=1.5)

        for h, test_element in enumerate(test_elements):
            coefs = np.zeros(n)
            coefs[test_element] = 1
            for l in range(mesh.elements.shape[1]):
                element = mesh.elements[:, l]
                xs, ns, dxns = eval_func(l, coefs, element, param_map, space, ref_data)

                axs[h][0].plot(xs, ns, *Plotter.args, **Plotter.kwargs)
                axs[h][1].plot(xs, dxns, *Plotter.args, **Plotter.kwargs)

            axs[h][0].set_xlabel("$x$")
            axs[h][0].set_ylabel("$N(x)$")
            axs[h][0].grid()

            axs[h][1].set_xlabel("$x$")
            axs[h][1].set_ylabel("$N'(x)$")
            axs[h][1].grid()
        plt.savefig(Plotter.add_folder(filename), bbox_inches="tight")
