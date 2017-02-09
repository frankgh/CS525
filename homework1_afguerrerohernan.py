import numpy as np


def problem1(A, B):
    """
    Given matrices A and B, compute and return an expression for A + B
    :param A: matrix A
    :param B: matrix B
    :return: A + B
    """
    return A + B


def problem2(A, B, C):
    """
    Given matrices A, B, and C, compute and return AB - C (i.e.,
    right-multiply matrix A by matrix B, and then subtract C). Use dot
    or numpy.dot
    :param A: matrix A
    :param B: matrix B
    :param C: matrix C
    :return: AB - C
    """
    return A.dot(B) - C


def problem3(A, B, C):
    """
    Given matrices A, B, and C, return A (.) B + C(T), where (.)
    represents the element-wise (Hadamard) product and T represents
    matrix transpose. In numpy, the element-wise product is obtained
    simply with *.
    :param A: matrix A
    :param B: matrix B
    :param C: matrix C
    :return: A (.) B + C(T)
    """
    return np.add(np.array(A) * np.array(B), C.transpose())


def problem4(x, y):
    """
    Given column vectors x and y, compute the inner product of x and y (i.e, x(T) y)
    :param x: column vector x
    :param y: column vector y
    :return: x(T) y
    """
    return np.dot(x.transpose(), y)


def problem5(A):
    """
    Given matrix A, return a matrix with the same dimensions as A but that contains
    all zeros. Use numpy.zeros
    :param A: matrix A
    :return: a matrix of same dimensions as A with all zeroes
    """
    return np.zeros(A.shape)


def problem6(A):
    """
    Given matrix A, return a vector with the same number of rows as A but that
    contains all ones. Use numpy.ones
    :param A: matrix A
    :return: a vector with the same number of rows as A but that contains all ones
    """
    return np.ones((A.shape[0], 1))


def problem7(A):
    """
    Given (invertible) matrix A, compute A(-1)
    :param A: matrix A
    :return: the inverse of matrix A
    """
    return np.linalg.inv(A)


def problem8(A, x):
    """
    Given a square matrix A and column vector x, use numpy.linalg.solve to
    compute A(-1) x.
    :param A: matrix A
    :param x: column vector x
    :return: A(-1) x
    """
    return np.linalg.solve(A, x)


def problem9(A, x):
    """
    Given square matrix A and row vector x, use numpy.linalg.solve to compute
    x A(-1).
    :param A: matrix A
    :param x: row vector x
    :return: xA(-1)
    """
    return np.linalg.solve(A.transpose(), x.transpose())


def problem10(A, alpha):
    """
    Given square matrix A and (scalar) alpha, compute A + alpha * I,
    where I is the identity matrix with the same dimensions as A.
    Use numpy.eye
    :param A: square matrix A
    :param alpha: scalar alpha
    :return: A + alpha * I
    """
    return A + alpha * np.eye(A.shape[0])


def problem11(A, i, j):
    """
    Given matrix A and integers i, j, return the jth column of
    the ith row of A, i.e., Aij .
    :param A: matrix A
    :param i: ith row
    :param j: jth column of the ith row
    :return: Aij
    """
    return A[i, j]


def problem12(A, i):
    """
    Given matrix A and integer i, return the sum of all the entries
    in the ith row, i.e., Pj Aij . Do not use a loop, which in Python
    is very slow. Instead use the numpy.sum function.
    :param A: matrix A
    :param i: integer i
    :return: sum of all the entries in the ith row
    """
    return np.sum(A[i:i + 1])


def problem13(A, c, d):
    """
    Given matrix A and scalars c, d, compute the arithmetic mean over
    all entries of A that are between c and d (inclusive). In other words,
    if S = {(i, j) : c <= Aij <= d}, then compute 1 |S| P(i,j) S Aij .
    Use numpy.nonzero along with numpy.mean.
    :param A:
    :param c:
    :param d:
    :return:
    """
    b = A[(A >= c).nonzero()]
    return np.mean(b[(b <= d).nonzero()])


def problem14(A, k):
    """
    Given an (n x n) matrix A and integer k, return an (n x k) matrix containing
    the right-eigenvectors of A corresponding to the k largest eigenvalues
    of A. Use numpy.linalg.eig to compute eigenvectors.
    :param A: matrix A
    :param k: integer k
    :return: (n x k) matrix containing the right-eigenvectors of A
                corresponding to the k largest eigenvalues of A
    """
    w, v = np.linalg.eig(A)
    return np.fliplr(v[:, -k:])


def problem15(x, k, m, s):
    """
     Given a n-dimensional column vector x, an integer k, and positive scalars m, s,
     return an (n x k) matrix, each of whose columns is a sample from multidimensional
     Gaussian distribution N (x + mz, sI), where z is an n-dimensional column vector
     containing all ones and I is the identity matrix.
     Use either numpy.random.multivariate_normal or numpy.random.randn. [ 5 pts ]
    :param x: column vector x
    :param k: integer k
    :param m: scalar m
    :param s: scalar s
    :return: an (n x k) matrix
    """
    mz = m * np.ones((len(x), 1))
    sI = s * np.eye(len(x))
    return np.random.multivariate_normal(np.add(x, mz).squeeze(), sI, k).transpose()
