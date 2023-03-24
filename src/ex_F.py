import numpy as np

from fem_students_1d import (
    create_mesh,
    create_param_map,
    create_fe_space,
    create_ref_data,
)
from utils.plotter import Plotter
from utils import eval_func
import matplotlib.pyplot as plt

coefs = np.loadtxt("data/coefficients.txt")

m = 5
p = 2
k = 1
L = 2
neval = 20
spacing_func = lambda i: i**2 * L / m**2

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

fig, axs = plt.subplots(1, 2)
fig.tight_layout()
for l in range(mesh.elements.shape[1]):
    element = mesh.elements[:, l]
    xs, ns, dxns = eval_func(
        l,
        coefs,
        element,
        param_map,
        evaluation_points,
        supported_bases,
        extraction_coefficients,
        reference_basis,
        reference_basis_derivatives,
    )

    axs[0].plot(xs, ns, "k")
    axs[1].plot(xs, dxns, "k")

axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$N(x)$")
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$N'(x)$")
plt.show()
