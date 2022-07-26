import SBD
import plotting
import numpy as np
from dataclasses import dataclass

def test_deconv_v1(levels: int, mes_size, ker_size, density, SNR: float) -> bool:
    Y, A, X = SBD.Y_factory(levels, mes_size, ker_size, density, SNR)
    A_solved, X_solved = SBD.deconvolve(Y, ker_size)
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
    bench_info = BenchmarkInfo(sample_num=1,
                               resolution=2,
                               max_defect_density=0.1,
                               min_defect_density=0.5 * (10 ** -4),
                               max_kernel_size=32,
                               min_kernel_size=8)
    if load:
        error_matrix = np.load(f'benchmark_{bench_info.resolution}_{bench_info.sample_num}.npy')
    else:
        error_matrix = SBD.benchmark(model='lista',
                                     defect_density_range=bench_info.defect_range(),
                                     kernel_size_range=bench_info.kernel_range(),
                                     samples=bench_info.sample_num)
        np.save(f'benchmark_{bench_info.resolution}_{bench_info.sample_num}', error_matrix)
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


