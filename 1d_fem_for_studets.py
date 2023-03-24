import numpy as np
from numpy.polynomial.legendre import leggauss as gaussquad
from scipy.interpolate import _bspl as bspl
import matplotlib.pyplot as plt

def create_ref_data(neval, deg, integrate=False):
    # reference unit domain
    reference_element = np.array([0, 1])
    if integrate is False:
        # point for plotting are equispaced on reference element
        x = np.linspace(reference_element[0], reference_element[1], neval)
        evaluation_points = x
        quadrature_weights = np.zeros((0,))
    else:
        # points (and weights) for integration are computed according to Gauss quadrature
        x, w = gaussquad(neval)
        evaluation_points = 0.5*(x + 1)
        quadrature_weights = w/2
    # knots for defining B-splines
    knt = np.concatenate((np.zeros((deg+1,),dtype=float),np.ones((deg+1,),dtype=float)),axis=0)
    # reference basis function values
    tmp = [bspl.evaluate_all_bspl(knt, deg, evaluation_points[i], deg, nu=0)
           for i in range(evaluation_points.shape[0])]
    reference_basis = np.vstack(tmp).T
    # reference basis function first derivatives
    tmp = [bspl.evaluate_all_bspl(knt, deg, evaluation_points[i], deg, nu=1)
           for i in range(evaluation_points.shape[0])]
    reference_basis_derivatives = np.vstack(tmp).T
    # store all data and return
    reference_data = {'reference_element': reference_element,
                      'evaluation_points': evaluation_points,
                      'quadrature_weights': quadrature_weights,
                      'deg': deg,
                      'reference_basis': reference_basis,
                      'reference_basis_derivatives': reference_basis_derivatives
    }
    return reference_data

def create_fe_space(deg, reg, mesh):
    def bezier_extraction(knt, deg):
        # breakpoints
        brk = np.unique(knt)
        # number of elements
        nel = brk.shape[0]-1
        # number of knots
        m = knt.shape[0]
        # assuming an open knotvector, knt[a] is the last repetition of the first knot
        a = deg
        # next knot
        b = a+1
        # Bezier element being processed
        nb = 0
        # first extraction matrix
        C = [np.eye(deg+1,deg+1, dtype=float)]
        # this is where knot-insertion coefficients are saved
        alphas = np.zeros((deg-1,),dtype=float)
        while b < m:
            # initialize extraction matrix for next element
            C.append(np.eye(deg+1,deg+1))
            # save index of current knot
            i = b
            # find last occurence of current knot
            while b < m-1 and knt[b+1] == knt[b]:
                b += 1
            # multiplicity of current knot
            mult = b-i+1
            # if multiplicity is < deg, smoothness is at least C0 and extraction may differ from an identity matrix
            if mult < deg:
                numer = knt[b] - knt[a]
                # smoothness of splines
                r = deg - mult
                # compute linear combination coefficients
                for j in range(deg-1,mult-1,-1):
                    alphas[j-mult] = numer / (knt[a+j+1]-knt[a])
                for j in range(r):
                    s = mult+j
                    for k in range(deg,s,-1):
                        alpha = alphas[k-s-1]
                        C[nb][:,k] = alpha*C[nb][:,k] + (1.0-alpha)*C[nb][:,k-1]
                    save = r-j
                    if b < m:
                        C[nb+1][save-1:j+save+1,save-1] = C[nb][deg-j-1:deg+1,deg]
            # increment element index
            nb += 1
            if b < m:
                a = b
                b += 1
            C = C[:nel]
        return C
    # number of mesh elements
    nel = mesh['m']
    # unique breakpoints
    if nel == 1:
        brk = mesh['elements'].T[0]
    else:
        brk = np.concatenate((mesh['elements'][0],
                              np.array([mesh['elements'][1][-1]])), axis=0)
    # multiplicity of each breakpoint
    mult = deg - reg
    # knot vector for B-spline definition
    knt = np.concatenate((np.ones((deg+1,), dtype=float) * brk[0],
                          np.ones((deg+1,), dtype=float) * brk[-1],
                          np.repeat(brk[1:-1],mult)), axis=0)
    knt = np.sort(knt)
    # coefficients of linear combination
    C = bezier_extraction(knt, deg)
    # dimension of finite element space
    dim = knt.shape[0]-deg-1
    # connectivity information (i.e., which bases are non-zero on which element)
    econn = np.zeros((nel,deg+1), dtype=int)
    for i in range(nel):
        if i == 0:
            econn[i] = np.arange( deg+1)
        else:
            econn[i] = econn[i-1] + mult
    # save and return
    space = {'n': dim,
             'supported_bases': econn,
             'extraction_coefficients': C
    }
    return space

def create_mesh(brk: np.ndarray) -> tuple[int, np.ndarray]:
    m = brk.shape[0]
    elements = np.zeros((m, 2))
    elements[0, :] = brk[:-1]
    elements[1, :] = brk[1:]

    return m, elements

def create_param_map(mesh):
    # IMPLEMENT HERE
    return mesh

def assemble_fe_problem(mesh, space ,ref_data, param_map, problem_B, problem_L, bc):
    # IMPLEMENT HERE
    return A, b

def problem_B(x,Nj,dNj,Nk,dNk):
    # IMPLEMENT HERE

def problem_L(x,Nj,dNj):
    # IMPLEMENT HERE


