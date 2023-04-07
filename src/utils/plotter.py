from typing import Any
import os

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """A class to wrap the plotting functions.

    (Static) Attributes
    -------------------
    _folder: str
        The folder to store the output figures.
    args: list[Any]
        The additional arguments for all plots.
    kwargs: dict[str, Any]
        The keyword arguments for all plots.
    """

    _folder: str = "figs"
    args: list[Any] = ["-ko"]
    kwargs: dict[str, Any] = {"markevery": [0, -1], "markersize": 2}

    @staticmethod
    def __clear__() -> None:
        """It clears the graphic objects."""

        plt.cla()
        plt.clf()

    @staticmethod
    def __setup_config__() -> None:
        """It sets up the matplotlib configuration."""

    plt.rc("text", usetex=True)
    plt.rcParams.update({"font.size": 11})

    @classmethod
    def add_folder(cls, path: str) -> str:
        """It adds the default folder to the input path.
        Parameters
        ----------
        path: str
            A path in string.
        Returns
        -------
        str
            The path with the added folder.
        """

        return os.path.join(cls._folder, path)

    @classmethod
    def plot_results(
        cls,
        xs_matrix: np.ndarray,
        ns_matrix: np.ndarray,
        dxns_matrix: np.ndarray,
        filename: str,
    ) -> None:
        """It plots the function and derivative evaluation.

        Parameters
        ----------
        xs_matrix: numpy.ndarray
            The matrix of evaluation points (columns) for each element (row).
        ns_matrix: numpy.ndarray
            The matrix of function evaluations at each point on the element.
        dxns_matrix: numpy.ndarray
            The matrix of derivatives of the function at each point on the element.
        filename: str
            The name of the file to write the figure to.
        """

        cls.__clear__()
        cls.__setup_config__()

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        fig.tight_layout(pad=2)
        m = xs_matrix.shape[0]

        for l in range(m):
            axs[0].plot(xs_matrix[l, :], ns_matrix[l, :], *cls.args, **cls.kwargs)
            axs[1].plot(xs_matrix[l, :], dxns_matrix[l, :], *cls.args, **cls.kwargs)

        axs[0].set_xlabel("$x$")
        axs[0].set_ylabel("$N(x)$")
        axs[0].grid()

        axs[1].set_xlabel("$x$")
        axs[1].set_ylabel("$N'(x)$")
        axs[1].grid()
        plt.savefig(cls.add_folder(filename), bbox_inches="tight")
