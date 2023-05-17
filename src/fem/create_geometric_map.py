import numpy as np

class create_geometric_map:

    def __init__(self, fe_geometry, ref_data) -> None:
        self.map = np.zeros(0)
        self.map_derivatives = np.zeros(0)
        self.imap_derivatives = np.zeros(0)
        self.__init_ref_data__(fe_geometry, ref_data)

    def __init_ref_data__(self, fe_geometry, ref_data) -> None:
        
        nqs = ref_data.neval ** 2
        self.map = np.zeros((nqs, 2, fe_geometry.m))
        self.map_derivatives = np.zeros((nqs, 4, fe_geometry.m))
        self.imap_derivatives = np.zeros((nqs, 4, fe_geometry.m))

        for i in range(fe_geometry.m):

            for j in range(2):
                self.map[:,j,i] = np.matmul(fe_geometry.map_coefficients[:,j,i],ref_data.reference_basis)
                for k in range(2):
                    self.map_derivatives[:,k * 2 + j,i] = np.matmul(fe_geometry.map_coefficients[:,j,i],ref_data.reference_basis_derivatives[:,:,k])
                    self.map_derivatives[:,k * 2 + j,i] = np.matmul(fe_geometry.map_coefficients[:,j,i],ref_data.reference_basis_derivatives[:,:,k])