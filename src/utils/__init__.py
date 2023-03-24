import numpy as np


def eval_func(
    l: int,
    coefs,
    element,
    param_map,
    evaluation_points,
    supported_bases,
    extraction_coefficients,
    reference_basis,
    reference_basis_derivatives,
):
    xs = param_map.func(evaluation_points, element[0], element[1])
    ns = np.zeros_like(xs)
    dxns = np.zeros_like(xs)
    for i, j in enumerate(supported_bases[l, :]):
        ej_i = extraction_coefficients[l][i, :]
        ns += coefs[j] * ej_i.dot(reference_basis)
        dxns += (
            coefs[j]
            * param_map.imap_derivatives[l]
            * ej_i.dot(reference_basis_derivatives)
        )

    return xs, ns, dxns
