import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pylops as pl
import random
from scipy import sparse
from autograd.scipy.signal import convolve as aconvolve
from scipy.signal import convolve
from numpy.random import default_rng
import autograd.numpy as anp
from numpy.linalg import norm
from pymanopt.manifolds import Sphere
from pymanopt import Problem
from pymanopt.solvers import TrustRegions
from model import ActivationNet, LISTA
from pathlib import Path
import torch
import os
from scipy import io
from tqdm import tqdm
from nanonispy.read import Grid
from dataclasses import dataclass


@dataclass
class Measurement:
    """
    This class represents the measurement (in the grid object) along with its activation map and kernel
    """
    grid: Grid
    kernel: np.ndarray = None
    activation_map: np.ndarray = None
    _level_list: list = None
    _kernel_shape: (int, int) = None

    @property
    def kernel_shape(self):
        return self._kernel_shape if self.kernel is None else np.shape(self.kernel)

    @kernel_shape.setter
    def kernel_shape(self, shape: (int, int)):
        """Changes the shape of the kernel and crops it to the new shape"""
        assert len(shape) == 2, 'Kernel shape has to be 2 numbers.'
        self._kernel_shape = shape
        if self.kernel is not None:
            self.kernel = crop_to_center(self.kernel, shape)

    @property
    def topo(self):
        return self.grid.signals['topo']

    @property
    def DoS(self):
        return np.moveaxis(self.grid.signals['LIX 1 omega (A)'], -1, 0)

    @property
    def DoS_over_I(self):
        return np.moveaxis(np.divide(self.grid.signals['LIX 1 omega (A)'], self.grid.signals['Current (A)']), -1, 0)

    @property
    def level_num(self):
        return np.shape(self.grid.signals['LIX 1 omega (A)'])[-1] if self._level_list is None else len(self._level_list)

    @property
    def level_list(self):
        return [*range(self.level_num)] if self._level_list is None else self._level_list

    @level_list.setter
    def level_list(self, given_list):
        assert np.all(np.isin(given_list, [*range(self.level_num), None])), \
            f'Some of the levels in {given_list} the given list don\'t exist in the measurement'
        self._level_list = given_list


def deconv_v0(Y, kernel_size, l_i, l_f, alpha):
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

    for _ in range(iteration_number):
        A_big, X = Asolve(A_big, sparse, Y, X)
        A_big, X = center(kernel_array, A, X)
        sparse = alpha * sparse

    # Output

    A_out = crop_to_center(A_big, kernel_array)
    X_out = FISTA(sparse, A_out, X)
    return A_out, X_out


def deconv_v1(filename: str, kernel_shape: (int, int), topography=False, level_list=None) -> (np.ndarray, np.ndarray):
    path = Path(filename)
    assert path.is_file(), f'There is no file here: {path}'
    assert path.with_suffix('.3ds'), 'The suffix must be .3ds'
    measurement = Measurement(Grid(path))
    measurement.kernel_shape = kernel_shape
    measurement.level_list = level_list

    # Solving
    X_solved = measurement_to_activation(measurement.topo if topography else measurement.DoS)
    A_solved = measurement_to_ker(measurement, X_solved)

    return A_solved, X_solved


def measurement_to_ker(measurement: Measurement, X):
    # Initial random guess
    kernel_rnd = np.random.normal(0, 1,tuple(2 * m for m in measurement.kernel_shape))
    kernel_rnd = sphere_norm_by_layer(kernel_rnd)
    # Solving
    kernel = np.zeros((measurement.level_num,) + tuple(2 * m for m in measurement.kernel_shape))
    for i, level in enumerate(measurement.level_list):
        if i == 1:
            kernel[i] = np.array(RTRM(1e-5, X, normalize_measurement(measurement.DoS[level]), kernel_rnd)[0])
        else:
            kernel[i] = np.array(RTRM(1e-5, X, normalize_measurement(measurement.DoS[level]), kernel[i - 1])[0])

    # Cropping and normalizing
    kernel = crop_to_center(kernel, measurement.kernel_shape)
    return sphere_norm_by_layer(kernel)


def RTRM(lam_in, X_in, Y, A0, verbose=0):
    """
    Minimizing A for cost_fun using Riemannian Trust-Region Method (RTRM) over the sphere.
    :param lam_in: Sparsity parameter
    :param X_in: Activation map
    :param Y: The data
    :param A0: Initial guess for A
    :return:
    """
    # 1. Initializing sphere manifold
    A_in = np.expand_dims(A0, 0) if np.ndim(A0) == 2 else A0
    Y_in = np.expand_dims(Y, 0) if np.ndim(A0) == 2 else Y
    s, m1, m2 = np.shape(A_in)
    sphere = Sphere(s, m1, m2)

    # 2. Defining the problem
    def cost(A): return cost_fun(lam_in, A, X_in, Y_in)

    problem = Problem(manifold=sphere, cost=cost)
    # 3. Solving
    solver = TrustRegions(mingradnorm=5e-6, maxtime=1.5 * 60, logverbosity=verbose)
    A_out = solver.solve(problem, x=A_in)
    return A_out


def FISTA(lam_in, A_in, Y, niter=1):
    s, n1, n2 = np.shape(Y)
    s, m1, m2 = np.shape(A_in)

    Cop = pl.signalprocessing.ConvolveND(N=s * n1 * n2, h=A_in, dims=(s, n1, n2), offset=(0, m1 // 2, m2 // 2),
                                         dirs=[1, 2])
    X = pl.optimization.sparsity.FISTA(Cop, Y.flatten(), niter, eps=2 * lam_in, threshkind='hard')[0]

    X = X.reshape((s, n1, n2))
    return np.median(X, axis=0)


def cost_fun(lambda_in, A, X, Y):
    """
    The cost function = 0.5|A conv X - Y|**2 + lambda * r(X)
    :param lambda_in:
    :param A:
    :param X:
    :param Y:
    :return: A real number
    """
    sX = sparse.csr_matrix(X)
    s, n1, n2 = np.shape(Y)
    A_con_X = aconvolve(sX.A, A, mode='full', axes=([0, 1], [1, 2]))
    A_con_X = crop_to_center(A_con_X, (n1, n2))
    return 0.5 * anp.sum((A_con_X - Y) ** 2) + lambda_in * regulator(sX.A)


def regulator(X):
    """
    The pseudo-Huber regulator
    :param X: 2D matrix
    :return: A real number
    """
    mu = 10 ** -6  # A small positive number (chosen in the paper to be 10 ** -6)
    return np.sum(mu ** 2 * (anp.sqrt(1 + (mu ** -2) * X) - 1))


def Asolve(A_in, lambda_in, Y, X=None):
    X_in = FISTA(lambda_in, A_in, Y) if X is None else X
    A_out = RTRM(lambda_in, X_in, Y, A_in)
    X_out = FISTA(lambda_in, A_in, Y)
    raise NotImplementedError


def ndarray_to_tensor(array):
    """
    Turns an ndarray to a torch tensor
    """
    return torch.tensor(array.astype(np.float)).unsqueeze(dim=0)


def measurement_to_activation(measurement, model='lista'):
    """
    Takes in n1 by n2 torch tensor of a measurement and returns an n1 by n2 torch tensor of its activation map
    """
    assert np.ndim(measurement) in [2, 3], f'Your measurement has {np.ndim(measurement)} dimensions'
    if model == 'cnn':
        net = ActivationNet()
        trained_model_path = Path('trained_model_norm.pt', map_location=torch.device('cpu'))
    elif model == 'lista':
        net = LISTA(5, iter_num=10)
        trained_model_path = Path('trained_lista_5layers.pt', map_location=torch.device('cpu'))
    if trained_model_path.is_file():
        net.load_state_dict(torch.load(trained_model_path))
        print('Loaded parameters from your trained model.')
    else:
        print('No trained model detected.')
    net.eval()
    net.cpu()
    net.double()
    mes_shape = np.shape(measurement)

    if np.ndim(measurement) == 3:
        activations = np.zeros(mes_shape)

        for level in range(mes_shape[0]):
            temp_mes = measurement[level]
            temp_mes = temp_mes[np.newaxis, :, :]
            measurement_tensor = ndarray_to_tensor(temp_mes)
            temp_act = net(measurement_tensor)[0][0].data.numpy()

            # This next bit is a filter to zero out all the small numbers in the map
            # temp_act[temp_act < 10 * np.mean(temp_act)] = 0
            temp_act = temp_act / np.sum(temp_act)
            activations[level] = temp_act

        activation = np.mean(activations, axis=0)

    else:
        measurement_tensor = ndarray_to_tensor(np.expand_dims(measurement, 0))
        activation = net(measurement_tensor)[0][0].data.numpy()

    activation[activation < np.max(activation) / 100] = 0
    activation = activation / np.sum(activation)
    # activation = net(ndarray_to_tensor(np.expand_dims(activation, 0)))[0][0].data.numpy()
    return activation


def Y_factory(s, Y_size, A_size, density, SNR=0):
    """
    This function produces a QPI measurement with specified size, defect density, kernel size and number of levels
    """
    n1, n2 = Y_size
    m1, m2 = A_size
    A = kernel_factory(s, m1, m2)
    X = sparse.random(n1, n2, density)
    X = X / np.sum(X)
    Y = np.zeros([s, n1, n2])

    for level in range(s):
        Y[level] = convolve(X.A, A[level], mode='same')
        eta = np.var(Y[level]) / SNR
        noise = np.random.normal(0, np.sqrt(eta), (n1, n2))
        Y[level] += noise

    return Y, A, X


def kernel_factory(s: int, m1: int, m2: int):
    """
    This function produces a set of s random m1 by m2 kernels
    """
    m_max = max(m1, m2)
    A = np.zeros([s, m_max, m_max], dtype=float)
    symmetry = random.choice([2, 3, 4, 6])
    half_sym = np.floor(symmetry / 2).astype('int')
    lowest_k = 0.5
    highest_k = 3
    k = np.random.uniform(lowest_k, highest_k, [s, symmetry])
    x, y = np.meshgrid(np.linspace(-1, 1, m_max), np.linspace(-1, 1, m_max))
    arb_angle = np.random.uniform(0, 2 * np.pi)

    for direction in range(symmetry):
        ang = direction * 180 / symmetry
        ang = arb_angle + ang * np.pi / 180
        r = (x * np.cos(ang) + np.sin(ang) * y)
        phi = np.random.uniform(0, 2 * np.pi)
        for i in range(s):
            # Adding normal decay
            sigma = np.random.uniform(0.2, 0.5)
            decay = gaussian_window(m_max, m_max, sigma)
            A[i, :, :] += np.cos(2 * np.pi * k[i, direction % half_sym] * r) * decay

    # Normalizing:
    A = np.abs(A)
    A = sphere_norm_by_layer(A)
    return A


def sphere_norm_by_layer(M):
    """
    Returns your matrix normalized to the unit sphere.
    :param M: A matrix with 3 dimensions
    :return: M normalized such that the sum of it's elements squared is one.
    """
    assert len(np.shape(M)) == 3, 'The matrix does not have 3 dim'
    return M / norm(M, axis=(-2, -1))[:, None, None]


def gaussian_window(n1, n2, sig=1, mu=0):
    """
    This function produces a 2D gaussian of size n1 by n2
    """
    x, y = np.meshgrid(np.linspace(-1, 1, n1), np.linspace(-1, 1, n2))
    d = np.sqrt(x * x + y * y)
    return np.exp(-((d - mu) ** 2 / (2.0 * sig ** 2)))


def center(size, A, X):
    """
    This function shifts the kernel (A) to center around its max sub-matrix, and shifts the activation map (X) to the
    opposite direction.
    """
    n1, n2 = np.shape(A)[-2], np.shape(A)[-1]
    s1, s2 = size[-2], size[-1]
    center_corner1 = np.floor((n1 - s1) / 2)
    center_corner2 = np.floor((n2 - s2) / 2)
    max_corner = max_submatrix_pos(A, size[-2], size[-1])
    A_shift = tuple([np.floor(center_corner1 - max_corner[0]).astype('int'),
                     np.floor(center_corner2 - max_corner[1]).astype('int')])
    X_shift = tuple(- j for j in A_shift)
    centered_A = shift(A, A_shift)
    centered_X = shift(X, X_shift)

    return centered_A, centered_X


def shift(mt, s=(0, 0)):
    """
    Takes a matrix and a tuple and returns that matrix shifted in the last 2 axes according to the tuple.
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
    """
    Crops a (s, n1, n2) matrix to a (s, m1, m2) matrix around the center
    :param M: dim=3 matrix
    :param box_shape: (m1, m2) or (s, m1, m2)
    :return: dim=3 matrix
    """
    if len(box_shape) == 3:
        diff = np.subtract(np.shape(M)[1:], box_shape[1:])
        gap = diff // 2
        return M[:, gap[0]:gap[0] + box_shape[1], gap[1]:gap[1] + box_shape[2]]
    diff = np.subtract(np.shape(M)[1:], box_shape)
    gap = diff // 2
    return M[:, gap[0]:gap[0] + box_shape[0], gap[1]:gap[1] + box_shape[1]]


def recovery_error(A_guess, A_true):
    """
    Returns the recovery error defined by: (2/pi) * arccos|<A_g, A_t>|
    """
    return (2 / np.pi) * np.arccos(np.round(abs(np.dot(A_guess.flatten(), A_true.flatten())), 10))


def benchmark(model, defect_density_range, kernel_size_range, SNR=2, samples=20):
    """
    Returns a recovery error matrix
    """
    error_matrix = np.zeros([len(defect_density_range), len(kernel_size_range)])
    for i, defect_density in enumerate(tqdm(defect_density_range)):
        print(f'loop {i + 1} out of {len(defect_density_range)} \n')
        for j, kernel_size in enumerate(kernel_size_range):
            errors = np.zeros(samples)
            for sample in range(samples):
                Y, A, X = Y_factory(1, (256, 256), (kernel_size, kernel_size), defect_density, SNR)

                X_guess = measurement_to_activation(Y, model=model)

                A_rand = np.random.normal(0, 1, (1, 2 * kernel_size, 2 * kernel_size))
                A_rand = A_rand / np.linalg.norm(A_rand)

                A_solved = RTRM(1e-5, X_guess, Y, A_rand)
                A_solved = crop_to_center(A_solved, (1, kernel_size, kernel_size))
                A_solved = A_solved / np.linalg.norm(A_solved)  # norm = 1

                errors[sample] = recovery_error(A_solved[0], A[0])
            error_matrix[i, j] = np.median(errors)
    return error_matrix


def compare_fft_plot(matrix1, matrix2, title):
    bigger_shape = np.shape(matrix1) if np.shape(matrix1)[1] > np.shape(matrix2)[1] else np.shape(matrix2)
    fft_matrix1 = np.fft.fftshift(np.fft.fft2(matrix1, s=bigger_shape))
    fft_matrix2 = np.fft.fftshift(np.fft.fft2(matrix2, s=bigger_shape))

    # Real Part
    fig, ax = plt.subplots(2, 3, figsize=(18, 6))
    fig.suptitle(title)

    ax[0, 0].imshow(np.real(fft_matrix1), cmap='bwr', norm=colors.CenteredNorm())
    ax[0, 0].set_title('Real Part 1')
    ax[0, 1].imshow(np.imag(fft_matrix1), cmap='bwr', norm=colors.CenteredNorm())
    ax[0, 1].set_title('Imaginary Part 1')
    ax[0, 2].imshow(np.abs(fft_matrix1), cmap='bwr', norm=colors.CenteredNorm())
    ax[0, 2].set_title('Absolute Value 1')

    ax[1, 0].imshow(np.real(fft_matrix2), cmap='bwr', norm=colors.CenteredNorm())
    ax[1, 0].set_title('Real Part 2')
    ax[1, 1].imshow(np.imag(fft_matrix2), cmap='bwr', norm=colors.CenteredNorm())
    ax[1, 1].set_title('Imaginary Part 2')
    ax[1, 2].imshow(np.abs(fft_matrix2), cmap='bwr', norm=colors.CenteredNorm())
    ax[1, 2].set_title('Absolute Value 2')
    for j in range(2):
        for i in range(3):
            ax[j, i].set_axis_off()
    plt.tight_layout()
    plt.show()


def save_data(number_of_samples, measurement_size, kernel_size, SNR=2, training=False, validation=False, testing=False):
    files_in_folder = os.listdir()
    if training and 'training_dataset' not in files_in_folder:
        os.system("mkdir training_dataset")
    if validation and 'validation_dataset' not in files_in_folder:
        os.system("mkdir validation_dataset")
    if testing and 'testing_dataset' not in files_in_folder:
        os.system("mkdir testing_dataset")

    density_exponent = np.random.uniform(low=-3.5, high=-1.5, size=(number_of_samples,))

    E, n1, n2 = measurement_size
    for i in range(number_of_samples):
        # sample_ker_size = int(np.random.uniform(kernel_size[0]/2, kernel_size[0]*2))
        sample_SNR = np.random.uniform(SNR / 2, 5 * SNR)
        temp_measurement, temp_kernel, temp_activation_map = Y_factory(E, (n1, n2),
                                                                       kernel_size,
                                                                       10 ** density_exponent[i],
                                                                       sample_SNR)
        if training:
            np.save(os.getcwd() + '/training_dataset/kernel_%d' % i, temp_kernel)
            np.save(os.getcwd() + '/training_dataset/measurement_%d' % i, temp_measurement)
            io.mmwrite(os.getcwd() + '/training_dataset/activation_%d' % i, temp_activation_map)
        elif validation:
            np.save(os.getcwd() + '/validation_dataset/kernel_%d' % i, temp_kernel)
            np.save(os.getcwd() + '/validation_dataset/measurement_%d' % i, temp_measurement)
            io.mmwrite(os.getcwd() + '/validation_dataset/activation_%d' % i, temp_activation_map)
        elif testing:
            np.save(os.getcwd() + '/testing_dataset/kernel_%d' % i, temp_kernel)
            np.save(os.getcwd() + '/testing_dataset/measurement_%d' % i, temp_measurement)
            io.mmwrite(os.getcwd() + '/testing_dataset/activation_%d' % i, temp_activation_map)
    if not training and not validation and not testing:
        print("Specify validation or training to save files.")


def normalize_measurement(Y):
    """
    Normalizes the input such that mean[Y]=0 and STD[Y]=0.0025
    """
    return (0.0025 / np.std(Y)) * (Y - np.mean(Y))
# measurement_shape = (1, 128, 128)
# kernel_shape = (16, 16)
#
# # save_data(10000, measurement_shape, kernel_shape, training=True)
# # save_data(2000, measurement_shape, kernel_shape, validation=True)
# save_data(100, (1, 200, 200), (20, 20), testing=True)
