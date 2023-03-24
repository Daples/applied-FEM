from fem_students_1d import (
    create_mesh,
    create_param_map,
    create_fe_space,
    create_ref_data,
)

import numpy as np
import matplotlib.pyplot as plt

m = 4
p = 1
k = 0
L = 2
Omega = [0, L]
brk = np.linspace(0, L, m + 1)

mesh = create_mesh(brk)
param_map = create_param_map(mesh)
space = create_fe_space(p, k, mesh)
n = space["n"]
supported_bases = space["supported_bases"]
extraction_coefficients = space["extraction_coefficients"]

neval = 10
deg = p
integrate_flag = False
ref_data = create_ref_data(neval, deg, integrate_flag)
reference_basis = ref_data["reference_basis"]
reference_basis_derivatives = ref_data["reference_basis_derivatives"]
evaluation_points = ref_data["evaluation_points"]

fig, axs = plt.subplots(3, 2)
for h, j in enumerate([0, 3, n - 1]):
    arr = np.zeros((3, m * neval))
    for l in range(mesh.elements.shape[1]):
        element = mesh.elements[:, l]
        lower_index = l * neval
        upper_index = lower_index + neval
        print(f"({lower_index}, {upper_index})")
        arr[0, lower_index:upper_index] = param_map.func(
            evaluation_points, element[0], element[1]
        )

        aux = supported_bases[l] == j
        is_supported = aux.any()

        if is_supported:
            index_support = np.where(aux)[0][0]
            ej_i = extraction_coefficients[l][index_support, :]
            arr[1, lower_index:upper_index] = ej_i.dot(reference_basis)
            arr[2, lower_index:upper_index] = param_map.imap_derivatives[l] * ej_i.dot(
                reference_basis_derivatives
            )

    axs[h, 0].plot(arr[0], arr[1], label=f"{j}")
    axs[h, 1].plot(arr[0], arr[2], label=f"{j}")
plt.show()
