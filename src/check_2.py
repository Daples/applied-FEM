from utils import read_mat
from fem.create_geometric_map import create_geometric_map
from fem.create_ref_data import create_ref_data
import numpy as np 
import matplotlib.pyplot as plt
from fem.assembler import Assembler


problem_B = lambda x, Nj, dxNj, dyNj, Nk, dxNk, dyNk: np.multiply(dxNj, dxNk) + np.multiply(dyNj, dyNk)
problem_L = lambda x, Nj, dxNj, dyNj: Nj 

fe_geometry, fe_space = read_mat("data/distressed_robotD.mat")

ref_data = create_ref_data(3, [2, 2], True)
geom_map = create_geometric_map(fe_geometry, ref_data)

A, b = Assembler.two_dimensional(fe_space,
        ref_data,
        geom_map,
        fe_geometry,
        problem_B,
        problem_L)


u = np.solve(A,b)




