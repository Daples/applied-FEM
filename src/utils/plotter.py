from typing import Any, Callable
import os

import matplotlib.pyplot as plt

from utils._typing import DataArray
from utils.data_handler import DataHandler


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
        xs_matrix: DataArray,
        ns_matrix: DataArray,
        dxns_matrix: DataArray,
        filename: str,
        result_label: str = "N",
        x_exact: DataArray | None = None,
        exact_solution: DataArray | None = None,
        d_exact_solution: DataArray | None = None,
    ) -> None:
        """It plots the function and derivative evaluation.

        Parameters
        ----------
        xs_matrix: utils._typing.DataArray
            The matrix of evaluation points (columns) for each element (row).
        ns_matrix: utils._typing.DataArray
            The matrix of function evaluations at each point on the element.
        dxns_matrix: utils._typing.DataArray
            The matrix of derivatives of the function at each point on the element.
        filename: str
            The name of the file to write the figure to.
        result_label: str, optional
            The result ylabel on the plot. Default: "N".
        x_exact: utils._typing.DataArray | None, optional
            The evaluation points of the exact solution. Default: None
        exact_solution: utils._typing.DataArray | None, optional
            The exact solution of the problem. Default: None
        d_exact_solution: utils._typing.DataArray | None, optional
            The derivatives of the exact solution at the evaluation points.
        """

        cls.__clear__()
        cls.__setup_config__()
        xs_matrix, ns_matrix, dxns_matrix = DataHandler.__cast_array__(
            xs_matrix, ns_matrix, dxns_matrix
        )
        if (
            x_exact is not None
            and exact_solution is not None
            and d_exact_solution is not None
        ):
            x_exact, exact_solution, d_exact_solution = DataHandler.__cast_array__(
                x_exact, exact_solution, d_exact_solution
            )

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
        x: DataArray,
        y: DataArray,
        path: str,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
    ) -> None:
        """It creates a plot with standard formatting.

        Parameters
        ----------
        x: utils._typing.DataArray
            The data on horizontal axis.
        y: utils._typing.DataArray
            The data on vertical axis.
        path: str
            The name to save the figure with.
        xlabel: str, optional
            The label of the horizontal axis.
        ylabel: str, optional
            The label of the vertical axis.
        """

        cls.__clear__()
        cls.__setup_config__()
        x, y = DataHandler.__cast_array__(x, y)

        plt.plot(x, y, "k-o", markersize=4)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.savefig(cls.add_folder(path), bbox_inches="tight")

    @classmethod
    def get_log_plot(
        cls,
        x: DataArray,
        y: DataArray,
        path: str,
        conv_order: Callable[[float], float],
        legend1: str,
        legend2: str,
        xlabel: str = "$h$",
        ylabel: str = "",
    ) -> None:
        """Create a log-log plot for convergence order checking.

        Parameters
        ----------
        x: utils._typing.DataArray
            The data on the horizontal axis.
        y: utils._typing.DataArray
            The data on the vertical axis.
        path: str
            The name to save the figure with.
        conv_order: float -> float
            The function that evaluates the exact convergence order.
        legend1: str
            The legend of the first line.
        legend2: str
            The legend of the second line.
        xlabel: str
            The label for the horizontal axis.
        xlabel: str
            The label for the vertical axis.
        """

        cls.__clear__()
        cls.__setup_config__()
        x, y = DataHandler.__cast_array__(x, y)

        _, ax = plt.subplots(1, 1)
        ords = [conv_order(i) for i in x]
        x = [1 / i for i in x]
        ax.loglog(x, y, label=legend1)
        ax.loglog(x, ords, "--*", label=legend2)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.savefig(cls.add_folder(path), bbox_inches="tight")
