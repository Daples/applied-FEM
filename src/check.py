from utils import read_mat
from fem.create_geometric_map import create_geometric_map
from fem.create_ref_data import create_ref_data


fe_geometry, _ = read_mat("data/star3.mat")

ref_data = create_ref_data(3, [2, 2], True)
geom_map = create_geometric_map(fe_geometry, ref_data)


print(geom_map.map)
print(geom_map.map_derivatives)
print(geom_map.imap_derivatives)