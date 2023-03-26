import os

import matplotlib.pyplot as plt


class Plotter:
    """A class to wrap the plotting functions.
    (Static) Attributes
    -------------------
    _levels: int
        The numer of levels for a countour plot.
    _folder: str
        The folder to store the output figures.
    """

    _folder: str = "figs"

    @staticmethod
    def __clear__() -> None:
        """It clears the graphic objects."""

        plt.cla()
        plt.clf()

    @staticmethod
    def __setup_config__() -> None:
        """It sets up the matplotlib configuration."""

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
