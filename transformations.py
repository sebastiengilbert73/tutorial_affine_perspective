import numpy as np

class Perspective():
    def __init__(self, correspondences_dict):
        if len(correspondences_dict) < 4:
            raise ValueError("transformations.Perspective.__init__(): len(correspondences_dict) ({}) < 4".format(len(correspondences_dict)))
        """
        | -X    -Y      -1      0       0       0       Xx      Yx      x | | A |   | 0 |
        |  0     0       0     -X      -Y      -1       Xy      Yy      y | | B | = | 0 |
        ....                                                                | C |
                                                                             ...
                                                                            | I |
        """

        A = np.zeros((2 * len(correspondences_dict), 9), np.float32)
        row = 0
        for xy, XY in correspondences_dict.items():
            x = xy[0]
            y = xy[1]
            X = XY[0]
            Y = XY[1]
            A[row, 0] = -X
            A[row, 1] = -Y
            A[row, 2] = -1
            A[row, 6] = X*x
            A[row, 7] = Y*x
            A[row, 8] = x
            A[row + 1, 3] = -X
            A[row + 1, 4] = -Y
            A[row + 1, 5] = -1
            A[row + 1, 6] = X*y
            A[row + 1, 7] = Y*y
            A[row + 1, 8] = y
            row += 2
        # Solve homogeneous system of linear equations
        # Cf. https://stackoverflow.com/questions/1835246/how-to-solve-homogeneous-linear-equations-with-numpy
        # Find the eigenvalues and eigenvector of A^T A
        e_vals, e_vecs = np.linalg.eig(np.dot(A.T, A))

        # Extract the eigenvector (column) associated with the minimum eigenvalue
        z = e_vecs[:, np.argmin(e_vals)]
        # Since the coefficients are defined up to a scale factor (we solved a homogeneous system of linear equations), we can multiply them by an arbitrary constant
        z = z/z[8]

        self.transformation_mtx = np.zeros((3, 3), np.float32)
        self.transformation_mtx[0, 0] = z[0]
        self.transformation_mtx[0, 1] = z[1]
        self.transformation_mtx[0, 2] = z[2]
        self.transformation_mtx[1, 0] = z[3]
        self.transformation_mtx[1, 1] = z[4]
        self.transformation_mtx[1, 2] = z[5]
        self.transformation_mtx[2, 0] = z[6]
        self.transformation_mtx[2, 1] = z[7]
        self.transformation_mtx[2, 2] = z[8]

class Affine():
    def __init__(self, correspondences_dict):
        if len(correspondences_dict) < 3:
            raise ValueError("transformations.Affine.__init__(): len(correspondeces_dict) ({}) < 3".format(len(correspondeces_dict)))
        """
        | X    Y   1   0   0   0 | | a00 |   | u |
        | 0    0   0   X   Y   1 | | a01 | = | v |
        | ...                    | | ... |   |...|
                                   | a12 |
        """
        A = np.zeros((2 * len(correspondences_dict), 6), np.float32)
        b = np.zeros((2 * len(correspondences_dict), 1), np.float32)
        row = 0
        for uv, XY in correspondences_dict.items():
            u = uv[0]
            v = uv[1]
            X = XY[0]
            Y = XY[1]
            A[row, 0] = X
            A[row, 1] = Y
            A[row, 2] = 1
            A[row + 1, 3] = X
            A[row + 1, 4] = Y
            A[row + 1, 5] = 1
            b[row] = u
            b[row + 1] = v
            row += 2
        # Solve overdetermined system of linear equations Ax = b
        x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
        self.transformation_mtx = np.zeros((2, 3), np.float32)
        self.transformation_mtx[0, 0] = x[0]
        self.transformation_mtx[0, 1] = x[1]
        self.transformation_mtx[0, 2] = x[2]
        self.transformation_mtx[1, 0] = x[3]
        self.transformation_mtx[1, 1] = x[4]
        self.transformation_mtx[1, 2] = x[5]