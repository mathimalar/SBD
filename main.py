from matplotlib import pyplot as plt
import numpy as np
import SBD
from SBD import Measurement, SimulationHandler, ThreeDSHandler, DeconvolvedMeasurement, deconv_v1
from dataclasses import dataclass
import plotting


def plot_fft_save(A, X, Y, level) -> None:
    """
    Plotting the measurement (big), the kernel and activation maps, and the measurements FFT along with the kernel FFT.
    """
    A_fft = np.fft.fftshift(np.fft.fft2(A[level], s=np.shape(Y[level])))
    Y_fft = np.fft.fftshift(np.fft.fft2(SBD.normalize_measurement(Y[level]), s=np.shape(Y[level])))

    fig = plt.figure(figsize=(12, 6))
    grid = plt.GridSpec(2, 4, wspace=0.2, hspace=0.3)

    Y_plot = fig.add_subplot(grid[:, :2])
    Y_plot.set_title(f'Measurement at slice: {level}')
    Y_plot.axis('off')

    Y_fft_plot = fig.add_subplot(grid[1, 2])
    Y_fft_plot.set_title('FFT(Y) Abs')
    Y_fft_plot.get_yaxis().set_visible(False)
    Y_fft_plot.get_xaxis().set_visible(False)

    A_fft_plot = fig.add_subplot(grid[1, 3])
    A_fft_plot.set_title('Kernel FFT')
    A_fft_plot.get_yaxis().set_visible(False)
    A_fft_plot.get_xaxis().set_visible(False)

    A_plot = fig.add_subplot(grid[0, 2])
    A_plot.set_title('Recovered kernel')
    A_plot.get_yaxis().set_visible(False)
    A_plot.get_xaxis().set_visible(False)

    X_plot = fig.add_subplot(grid[0, 3])
    X_plot.set_title('Activation map')
    X_plot.get_yaxis().set_visible(False)
    X_plot.get_xaxis().set_visible(False)

    Y_plot.imshow(Y[level], cmap='hot')

    Y_fft_im = Y_fft_plot.imshow(np.abs(Y_fft), cmap='afmhot')  # Should be logarithmic?
    A_fft_im = A_fft_plot.imshow(np.abs(A_fft), cmap='afmhot')  # Should be logarithmic?
    A_plot.imshow(A[level], cmap='hot')
    X_plot.imshow(X == 0, cmap='binary', interpolation='None')

    plt.colorbar(A_fft_im, ax=A_fft_plot)
    plt.colorbar(Y_fft_im, ax=Y_fft_plot)
    plt.tight_layout()
    plt.show()


def test_deconv_v1(levels: int, mes_size, ker_size, density, SNR: float) -> bool:
    Y, A, X = SBD.Y_factory(levels, mes_size, ker_size, density, SNR)
    A_solved, X_solved = SBD.deconv_v1(Y, ker_size)
    # test if the loss is better than some random thing
    raise NotImplementedError

    # count0 = np.count_nonzero(X.A)
    # count_guess = np.count_nonzero(X_guess)
    #
    # lista_error = SBD.recovery_error(A_solved[0], A[0])
    # side_by_side(A_solved[0], A[0], f'LISTA Kernels. Error = {np.round(lista_error, 4)}', 'Predicted', 'True')
    # SBD.compare_fft_plot(Y[0], A_solved[0], 'True FFT vs SBD FFT')


def test_benchmark(load=False) -> bool:
    """
    Uses the benchmark function from SBD and checks if it returns positive numbers within [0,1]
    """
    bench_info = BenchmarkInfo(sample_num=20,
                               resolution=20,
                               max_defect_density=0.5,
                               min_defect_density=0.5 * (10 ** -4),
                               max_kernel_size=62,
                               min_kernel_size=8)
    if load:
        error_matrix = np.load(f'benchmark_{bench_info.resolution}_{bench_info.sample_num}.npy')
    else:
        error_matrix = SBD.benchmark(model='lista',
                                     defect_density_range=bench_info.defect_range(),
                                     kernel_size_range=bench_info.kernel_range(),
                                     samples=bench_info.sample_num)
        error_matrix = np.save(f'benchmark_{bench_info.resolution}_{bench_info.sample_num}', error_matrix)
    plotting.plot_benchmark(error_matrix, bench_info.defect_range(), bench_info.kernel_range())
    return np.all(error_matrix >= 0) and np.all(error_matrix <= 1)


@dataclass
class BenchmarkInfo:
    """
    This represents all the information to make a benchmark test for the code for different parameter ranges
    """
    sample_num: int  # The number of samples to median over
    resolution: int  # The number of different defect densities and different kernel sizes
    max_defect_density: float  # As a ratio between the defects and non-defect pixels in a matrix
    min_defect_density: float  # As a ratio between the defects and non-defect pixels in a matrix
    max_kernel_size: int  # In pixels
    min_kernel_size: int  # In pixels

    def defect_range(self, logarithmic=True):
        if logarithmic:
            return np.logspace(np.log10(self.min_defect_density), np.log10(self.max_defect_density), self.resolution)
        else:
            return np.linspace(self.min_defect_density, self.max_defect_density, self.resolution)

    def kernel_range(self, logarithmic=False):
        if logarithmic:
            return np.logspace(np.log10(self.min_kernel_size), np.log10(self.max_kernel_size), self.resolution)
        else:
            return np.linspace(self.min_kernel_size, self.max_kernel_size, self.resolution)


def main():
    files = [r'T:\LT_data\TaAs\2015-12-08\Grid Spectroscopy001.3ds',
             r'T:\LT_data\TaAs\2016-01-01\Grid Spectroscopy002.3ds',
             r'T:\LT_data\TaAs\2016-01-04\Grid Spectroscopy002.3ds',
             r'T:\LT_data\Copper\2019-12-22\Grid Spectroscopy002.3ds',
             r'T:\LT_data\Copper\2014-03-20\Grid Spectroscopy002.3ds']

    kernel_size = (64, 64)

    handler1 = ThreeDSHandler(files[4], kernel_size)
    handler2 = SimulationHandler(5, (300, 300), kernel_size, SNR=20, defect_density=0.001)
    measurement = handler2.get_measurement_data()
    # deconv_measurement = deconv_v1(measurement)
    plotting.plot_fft(handler2.kernel, handler2.activation_map, measurement.density_of_states)


if __name__ == '__main__':
    main()

    # Testing max_submatrix_pos

    # m1, m2 = 2, 2
    # sample = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])  # np.random.randint(0, 100, size=(5, 30, 30))
    # i, j = SBD.max_submatrix_pos(sample, m1, m2)
    # flat_sample = np.sum(sample, axis=0)
    # plt.imshow(np.sum(sample, axis=0), interpolation='nearest')
    # plt.show()
    # print(flat_sample)
    # print(flat_sample[i: i + m2, j: j + m2])
    # print(f'i = {i}, j={j}')

    # Testing crop_to_center

    # kernel_shape = np.array([3, 6, 6])
    # a = np.random.randint(0, 10, size=kernel_shape)
    # padding = np.floor(kernel_shape[1:] / 2).astype('int')
    # padded = np.pad(a, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])))
    # print(f'padded = \n{padded}\ncropped = \n{SBD.crop_to_center(padded, kernel_shape)}')

    # Testing shift

    # arr = np.ones([3, 7, 7])
    # arr[:, 3:6, 1:4] = 20
    # c_arr, c_x = SBD.center((2, 2), arr, arr)
    # print(f'a =\n{arr}\n shifted =\n{c_arr}\n anti-shifted =\n{c_x}')

    # testing kernel_factory

    # sample_ker = SBD.kernel_factory(levels, 100, 100)
    # fig, ax = plt.subplots(1, levels)
    # fig.suptitle('Horizontally stacked subplots')
    # for i in range(levels):
    #     ax[i].imshow(sample_ker[i, :, :], cmap='hot', interpolation='nearest')
    # plt.show()

    # testing Y_factory
    # Y, A, X = SBD.Y_factory(levels, (n1, n2), (m1, m2), density, SNR)
    # A_noise = A + np.random.normal(0, A.mean() / 2, (levels, m1, m2))
    # A_noise = SBD.sphere_norm_by_layer(A_noise)
    #
    # fig, ax = plt.subplots(levels, 2)
    # fig.suptitle('Y:')
    # for i in range(levels):
    #     ax[i, 0].imshow(A[i, :, :], cmap='hot', interpolation='nearest')
    #     ax[i, 1].imshow(Y[i, :, :], cmap='hot', interpolation='nearest')
    #     for j in range(2):
    #         ax[i, j].set_xticks([])
    #         ax[i, j].set_yticks([])
    # fig.tight_layout()
    # fig.savefig('A_Y.jpg', dpi=500)
    #
    # fig2, ax2 = plt.subplots()
    # ax2.imshow(X.toarray(), cmap='hot', interpolation='nearest')
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    # fig2.savefig('X.jpg', dpi=500)
    # plt.show()

    # Testing cost function - WORKS
    # cost = SBD.cost_fun(0.05, A, X, Y)
    # print(cost)
    # cost = SBD.cost_fun(0.05, A_noise, X, Y)
    # print(cost)

    # Testing FISTA: - gives bad approximations for activation maps
    # Y, A, X = SBD.Y_factory(levels, (n1, n2), (m1, m2), density, SNR)
    # # A_noise = A + np.random.normal(0, A.mean() / 2, (levels, m1, m2))
    # # A_noise = SBD.sphere_norm_by_layer(A_noise)
    #
    # X_dense = X.toarray()
    # X_new = SBD.FISTA(1e-7, A, Y, niter=1000)
    # X_new = X_new / np.sum(X_new)
    #
    #
    # fig, ax = plt.subplots(1, 2)
    # fig.suptitle('Activation maps')
    # max_value = np.max(X)
    # ax[0].imshow(X_new, vmin=0, vmax=max_value, cmap='hot')
    # ax[0].set_title('X approx')
    # ax[1].imshow(X_dense, vmin=0, vmax=max_value, cmap='hot')
    # ax[1].set_title('X')
    # for i in range(2):
    #     ax[i].set_axis_off()
    # plt.savefig('x_approx', DPI=400)
    # plt.show()
    # error_X = np.sum(X - X_new) / np.sum(X)
    # print(f'X relative error: {round(100 * error_X)}%')

    # Testing conv operator: - works
    # Y, A, X = SBD.Y_factory(levels, (n1, n2), (m1, m2), density, SNR)
    # A_noise = A + np.random.normal(0, A.mean() / 2, (levels, m1, m2))
    # A_noise = SBD.sphere_norm_by_layer(A_noise)
    # Cop = pl.signalprocessing.ConvolveND(N=levels * n1 * n2, h=A, dims=(levels, n1, n2), offset=(0, m1 // 2, m2 // 2),
    #                                          dirs=(1, 2))
    # X_dense = X.toarray()
    # X_new = SBD.FISTA(0.5, A, Y, niter=50)
    # X_layers = np.array([X_dense for i in range(levels)])
    # A_conv_X = Cop * X_layers.flatten()
    # A_conv_X = A_conv_X.reshape((levels, n1, n2))
    # side_by_side(Y[0], A_conv_X[0], 'convolution test')
