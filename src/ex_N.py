import numpy as np

from fem_students_1d import (
    create_mesh,
    create_param_map,
    create_fe_space,
    create_ref_data,
    assemble_fe_problem,
)
from utils.plotter import Plotter
from utils import eval_func
import matplotlib.pyplot as plt

def problem_B(x, Nj, dNj, Nk, dNk):
    """"""

    return np.multiply(dNj, dNk)


def problem_L(x, Nj, dNj):
    """"""

    return (np.pi ** 2) * np.multiply(Nj, np.sin(np.pi * x))


m = 4
p = 2
k = 0
L = 2
neval = 3
spacing_func = lambda i: i * L / m
bc = [0, 1]

brk = np.array([spacing_func(i) for i in range(0, m + 1)])
mesh = create_mesh(brk)
param_map = create_param_map(mesh)
space = create_fe_space(p, k, mesh)
ref_data = create_ref_data(neval, p, True)

A, b = assemble_fe_problem(mesh, space, ref_data, param_map, problem_B, problem_L, bc)

np.savetxt("0/A.txt", A, delimiter=',', fmt = "%1.14f")
np.savetxt("0/b.txt", b, delimiter=',', fmt = "%1.14f")

u_sol = np.linalg.solve(A,b)
u_coefs = np.zeros(u_sol.shape[0]+2)
u_coefs[0] = bc[0]
u_coefs[-1] = bc[1]
u_coefs[1:-1] = u_sol

neval = 20
ref_data = create_ref_data(neval, p, True)

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
        u_coefs,
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

