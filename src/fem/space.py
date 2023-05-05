import numpy as np

from fem.mesh import Mesh


class Space:
    """A data class to represent the finite element space.

    Attributes
    ----------
    dim: int
        The dimension of the finite element space.
    supported_bases: numpy.ndarray
        The indices (columns) of the basis functions that are nonzero on the (row)
        element. Matrix of size `m x (p+1)`.
    extraction_coefficients: numpy.ndarray
        The coefficients of the reference basis functions on each element. Tensor of
        size `m x (p+1) x (p+1)`.
    """

    dim: int
    supported_bases: np.ndarray
    extraction_coefficients: np.ndarray

    def __init__(self, deg: int, reg: int, mesh: Mesh) -> None:
        self.dim: int = 0
        self.supported_bases: np.ndarray = np.zeros(0)
        self.extraction_coefficients: np.ndarray = np.zeros(0)
        self.__init_fe_space__(deg, reg, mesh)

    def __init_fe_space__(self, deg: int, reg: int, mesh: Mesh) -> None:
        """Initialize the finite element space. (Implemented by Deepesh Toshniwal)

        Parameters
        ----------
        deg: int
            The polynomial degree of the space.
        reg: int
            The smoothness requirement.
        mesh: fem.mesh.Mesh
            The mesh.
        """

        def bezier_extraction(knt: np.ndarray, deg: int) -> np.ndarray:
            # Breakpoints
            brk = np.unique(knt)
            # Number of elements
            nel = brk.shape[0] - 1
            # Number of knots
            m = knt.shape[0]
            # Assuming an open knotvector, knt[a] is the last repetition of the first knot
            a = deg
            # Next knot
            b = a + 1
            # Bezier element being processed
            nb = 0
            # First extraction matrix
            C = [np.eye(deg + 1, deg + 1, dtype=float)]

            # This is where knot-insertion coefficients are saved
            alphas = np.zeros((np.maximum(0, deg - 1),), dtype=float)
            while b < m:
                # Initialize extraction matrix for next element
                C.append(np.eye(deg + 1, deg + 1))
                # Save index of current knot
                i = b
                # Find last occurence of current knot
                while b < m - 1 and knt[b + 1] == knt[b]:
                    b += 1
                # Multiplicity of current knot
                mult = b - i + 1

                if mult < deg:
                    # Smoothness is at least C0 and extraction may differ from identity
                    numer = knt[b] - knt[a]
                    # Smoothness of splines
                    r = deg - mult
                    # Compute linear combination coefficients
                    for j in range(deg - 1, mult - 1, -1):
                        alphas[j - mult] = numer / (knt[a + j + 1] - knt[a])
                    for j in range(r):
                        s = mult + j
                        for k in range(deg, s, -1):
                            alpha = alphas[k - s - 1]
                            C[nb][:, k] = (
                                alpha * C[nb][:, k] + (1.0 - alpha) * C[nb][:, k - 1]
                            )
                        save = r - j
                        if b < m:
                            C[nb + 1][save - 1 : j + save + 1, save - 1] = C[nb][
                                deg - j - 1 : deg + 1, deg
                            ]
                # Increment element index
                nb += 1
                if b < m:
                    a = b
                    b += 1
                C = C[:nel]
            return np.array(C)

        # Number of mesh elements
        nel = mesh.m
        # Unique breakpoints
        if nel == 1:
            brk = mesh.elements.T[0]
        else:
            brk = np.concatenate(
                (mesh.elements[0], np.array([mesh.elements[1][-1]])), axis=0
            )
        # Multiplicity of each breakpoint
        mult = deg - reg
        # Knot vector for B-spline definition
        knt = np.concatenate(
            (
                np.ones((deg + 1,), dtype=float) * brk[0],
                np.ones((deg + 1,), dtype=float) * brk[-1],
                np.repeat(brk[1:-1], mult),
            ),
            axis=0,
        )
        knt = np.sort(knt)
        # Coefficients of linear combination
        C = bezier_extraction(knt, deg)
        # Dimension of finite element space
        dim = knt.shape[0] - deg - 1
        # Connectivity information (i.e., which bases are non-zero on which element)
        econn = np.zeros((nel, deg + 1), dtype=int)
        for i in range(nel):
            if i == 0:
                econn[i] = np.arange(deg + 1)
            else:
                econn[i] = econn[i - 1] + mult

        self.dim = dim
        self.supported_bases = econn
        self.extraction_coefficients = C
