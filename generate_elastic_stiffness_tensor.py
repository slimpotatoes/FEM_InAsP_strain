# Code to generate elastic stiffness tensor (Cijkl) adapted for FEniCS
# Sigma_ij = C_ijkl * epsilon_kl (4th order rank tensor in 3D space)

import numpy as np


class Generator(object):

    def __init__(self):
        self.symmetry = None
        self.stiff_tensor = np.zeros((3, 3, 3, 3))
        self.cijkl_list = []

    def import_elements(self, cijkl_list, symmetry):
        """Put he the cijkl in proper order (rising indices)"""
        self.symmetry = symmetry
        self.cijkl_list = cijkl_list
        if self.symmetry is 'cubic':
            if len(self.cijkl_list) != 3:
                print('Improper number of coefficient for cubic symmetry')
            else:
                for i in range(0, 3):
                    self.stiff_tensor[i, i, i, i] = self.cijkl_list[0]
                    for j in range(0, 3):
                        if j != i:
                            self.stiff_tensor[i, i, j, j] = self.cijkl_list[1]
                            self.stiff_tensor[i, j, i, j] = self.cijkl_list[2]
                            self.stiff_tensor[i, j, j, i] = self.cijkl_list[2]
                print('Elastic stiffness tensor imported')
        else:
            print('Non-cubic material not supported')

    def rotation_stiffness_tensor(self, P):
        """Taken from
        https://stackoverflow.com/questions/4962606/fast-tensor-rotation-with-numpy
        user : Philipp
        P : Rotation matrix = np.array(3x3)"""
        PP = np.outer(P, P)
        PPPP = np.outer(PP, PP).reshape(4 * P.shape)
        axes = ((0, 2, 4, 6), (0, 1, 2, 3))
        self.stiff_tensor = np.tensordot(PPPP, self.stiff_tensor, axes)

    def export_tensor(self, filename):
        np.save(filename, self.stiff_tensor)
        print("Tensor exported ==> ", filename)
