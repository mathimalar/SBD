import numpy as np
import pylops as pl
import random
from scipy import sparse
from scipy import signal
from matplotlib import pyplot as plt


def deconv(Y, kernel_size, l_i, l_f, alpha):
    """
    Preforms blind de-convolution given a measurement, kernel size, initial and final lambdas (sparsity), and decay rate.
    :param Y: measurement with shape (s, n1, n2)
    :param kernel_size: size of single real-space QPI, with shape (s, m1, m2)
    :param l_i: Initial sparsity constant. Bigger than l_f.
    :param l_f: Final sparsity constant. Bigger than l_f.
    :param alpha: Decay rate from l_i to l_f. Between zero and one.
    :return: returns the kernel and activation maps in the shapes of Y and kernel_size respectively.
    """

    assert l_i >= l_f, "l_i is smaller than l_f."

    # Initialization

    A = np.random.uniform(low=np.amin(Y), high=np.amax(Y), size=kernel_size)
    A, X = Asolve(A, l_i, Y)

    # Refinement

    kernel_array = np.array(kernel_size)
    pad = np.floor(kernel_array[1:] / 2)
    A_big = np.pad(A, ((0, 0), (pad[0], pad[0]), (pad[1], pad[1])))
    sparse = l_i
    iteration_number = np.ceil(np.log(l_f / l_i) / np.log(alpha))

    for i in range(iteration_number):
        A_big, X = Asolve(A_big, sparse, Y, X)
        A_big, X = center(kernel_array, A, X)
        sparse = alpha * sparse

    # Output

    A_out = crop_to_center(A_big, kernel_array)
    X_out = FISTA(sparse, A_out, X)
    return A_out, X_out


def FISTA(lam_in, A_in, Y):
    s, n1, n2 = np.shape(Y)
    s, m1, m2 = np.shape(A_in)
    Cop = pl.signalprocessing.ConvolveND(N=s*n1*n2, h=A_in, dims=(s, n1, n2), offset=(0, m1 // 2, m2 // 2), dirs=[1, 2])
    niter = 10  # Can be different!
    X, total_iter, cost = pl.optimization.sparsity.FISTA(Cop, Y, niter, 2 * lam_in)
    return X


def RTRM(lam_in, X_in, Y):
    raise NotImplementedError


def cost_fun(lambda_in, A, X, Y):
    sX = sparse.csr_matrix(X)
    s, m1, m2 = np.shape(A)
    A_con_X = np.zeros(s, m1, m2)
    for i in range(s):
        A_con_X[i] = signal.convolve2d(sX.A, A[i], mode='same')

    phi = 0.5 * np.sum((A_con_X - Y) ** 2) + lambda_in * np.sum(np.abs(X))
    return phi


def Asolve(A_in, lambda_in, Y, X=None):
    if X is None:
        X_in = FISTA(lambda_in, A_in, Y)
    else:
        X_in = X

    A_out = RTRM(lambda_in, X_in, Y)
    X_out = FISTA(lambda_in, A_in, Y)
    raise NotImplementedError


def Y_factory(s, Y_size, A_size, density, SNR=0):
    n1, n2 = Y_size
    m1, m2 = A_size
    A = kernel_factory(s, m1, m2)
    X = sparse.random(n1, n2, density)
    X = X / np.sum(X)
    Y = np.zeros([s, n1, n2])

    for level in range(s):
        Y[level] = signal.convolve2d(X.A, A[level], mode='same')
        eta = np.var(Y[level]) / SNR
        noise = np.random.normal(0, np.sqrt(eta), (n1, n2))
        Y[level] += noise

    return Y, A, X


def kernel_factory(s, m1, m2):
    m_max = max(m1, m2)
    A = np.zeros([s, m_max, m_max], dtype=complex)
    symmetry = random.choice([2, 3, 4, 6])
    half_sym = np.floor(symmetry / 2).astype('int')
    lowest_k = 0.5
    highest_k = 3
    k = np.zeros([s, symmetry])
    for level in range(s):
        k[level, :] = np.random.uniform(lowest_k, highest_k, symmetry)

    x, y = np.meshgrid(np.linspace(-1, 1, m_max), np.linspace(-1, 1, m_max))
    # dist = np.sqrt(x * x + y * y)
    # theta = np.arctan(x / y)
    arb_angle = np.random.uniform(0, 2 * np.pi)
    for direction in range(symmetry):
        ang = direction * 180 / symmetry
        ang = arb_angle + ang * np.pi / 180
        r = (x * np.cos(ang) + np.sin(ang) * y)
        for i in range(s):
            A[i, :, :] += np.cos(2 * np.pi * k[i, direction % half_sym] * r)

    # Adding decay
    sigma = np.random.uniform(0.3, 0.6)
    decay = gaussian_window(m_max, m_max, sigma)
    A = np.multiply(np.abs(A), decay)

    return A


def gaussian_window(n1, n2, sig=1, mu=0):
    x, y = np.meshgrid(np.linspace(-1, 1, n1), np.linspace(-1, 1, n2))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-((d - mu) ** 2 / (2.0 * sig ** 2)))
    return g


def center(size, A, X):
    n1, n2 = np.shape(A)[-2], np.shape(A)[-1]
    s1, s2 = size[-2], size[-1]
    center_corner1 = np.floor((n1 - s1) / 2)
    center_corner2 = np.floor((n2 - s2) / 2)
    max_corner = max_submatrix_pos(A, size[-2], size[-1])
    A_shift = tuple([np.floor(center_corner1 - max_corner[0]).astype('int'),
                     np.floor(center_corner2 - max_corner[1]).astype('int')])
    X_shift = tuple([(- j) for j in A_shift])
    centered_A = shift(A, A_shift)
    centered_X = shift(X, X_shift)

    return centered_A, centered_X


def shift(mt, s=(0, 0)):
    """
    Takes a matrix and a tuple and returns that matrix shifted according to the tuple.
    :param mt: matrix, shape len 2 or 3
    :param s: tuple defining the shift (row shift, column shift)
    :return: shifted matrix
    """
    z = np.zeros(np.shape(mt))
    s1, s2 = s
    row_shift = list(range(-s1, -s1 + z.shape[-2]))
    col_shift = list(range(-s2, -s2 + z.shape[-1]))
    for row in range(z.shape[-2]):
        for col in range(z.shape[-1]):
            in_bounds = z.shape[1] > row_shift[row] >= 0 and z.shape[2] > col_shift[col] >= 0
            if len(z.shape) == 2:
                z[row, col] = mt[row_shift[row], col_shift[col]] if in_bounds else 0
            elif len(z.shape) == 3:
                z[:, row, col] = mt[:, row_shift[row], col_shift[col]] if in_bounds else 0

    return z


def max_submatrix_pos(matrix, m1, m2):
    """
    Returns position of max submatrix.
    :param matrix: Matrix with shape (i, row, col)
    :param m1: rows in kernel
    :param m2: columns in kernel
    :return: row, column of max submatrix of size m1 by m2 in matrix
    """
    # Initializing
    sub_norm = 0
    pos1, pos2 = 0, 0
    arr = np.sum(np.array(matrix), axis=0)
    rows = arr.shape[0]
    cols = arr.shape[1]

    # Iterating
    for row in range(rows - m1 + 1):
        for col in range(cols - m2 + 1):
            running_sum = np.sum(arr[row:row + m1, col:col + m2])
            if running_sum > sub_norm:
                sub_norm = running_sum
                pos1, pos2 = row, col
    return np.array([pos1, pos2])


def crop_to_center(M, box_shape):
    diff = np.subtract(np.shape(M)[1:], box_shape[1:])
    gap = np.floor(diff / 2).astype('int')
    return M[:, gap[0]:gap[0] + box_shape[1], gap[1]:gap[1] + box_shape[2]]
