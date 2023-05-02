from typing import Any, Callable
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

    _folder: str = os.path.join(os.getcwd(), "figs")
    args: list[Any] = ["-o"]
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
        result_label: str = "N",
        x_exact: np.ndarray | None = None,
        exact_solution: np.ndarray | None = None,
        d_exact_solution: np.ndarray | None = None,
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
        result_label: str, optional
            The result ylabel on the plot. Default: "N".
        """

        cls.__clear__()
        cls.__setup_config__()

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        fig.tight_layout(pad=2)
        m = xs_matrix.shape[0]

        for l in range(m):
            kwargs = cls.kwargs | {"color": "k"}
            label = {}
            d_label = {}
            if l == 0:
                label = {"label": "$u_h$"}
                d_label = {"label": "$u_{h,x}$"}
            kwargs |= label
            d_kwargs = kwargs | d_label
            axs[0].plot(xs_matrix[l, :], ns_matrix[l, :], *cls.args, **kwargs)
            axs[1].plot(xs_matrix[l, :], dxns_matrix[l, :], *cls.args, **d_kwargs)

        if exact_solution is not None and d_exact_solution is not None:
            kwargs = cls.kwargs | {"color": "r"}
            axs[0].plot(
                x_exact,
                exact_solution,
                *cls.args,
                **kwargs,
                label="$u_e$",
            )
            axs[1].plot(
                x_exact,
                d_exact_solution,
                *cls.args,
                **kwargs,
                label="$u_{e,x}$",
            )

        xlabel = "$x$"
        ylabel = f"${result_label}(x)$"
        ylabel_p = f"${result_label}'(x)$"
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        axs[0].grid()

        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel_p)
        axs[1].grid()
        plt.legend()
        plt.savefig(cls.add_folder(filename), bbox_inches="tight")

    @classmethod
    def get_plot(
        cls,
        x: list[float],
        y: list[float],
        path: str,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
    ) -> None:
        """

        Parameters
        ----------
        """

        cls.__clear__()
        cls.__setup_config__()

        plt.plot(x, y, "k-o", markersize=4)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.savefig(cls.add_folder(path), bbox_inches="tight")

    @classmethod
    def get_log_plot(
        cls,
        x: list[float],
        y: list[float],
        path: str,
        conv_order: Callable,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
    ) -> None:
        """

        Parameters
        ----------
        """

        cls.__clear__()
        cls.__setup_config__()

        _, ax = plt.subplots(1, 1)
        ords = [conv_order(i) for i in x]
        x = [1 / i for i in x]
        ax.loglog(x, y, label="$||e||$")
        ax.loglog(x, ords, label="Conv")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.savefig(cls.add_folder(path), bbox_inches="tight")
