import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import SBD
from SBD import Measurement, Simulation, ThreeDSHandler, deconvolve
from dataclasses import dataclass
import plotting
import bench


def main():
    # # An example list of files to analyze
    # three_ds_paths = [r'T:\LT_data\TaAs\2015-12-08\Grid Spectroscopy001.3ds',
    #                   r'T:\LT_data\TaAs\2016-01-01\Grid Spectroscopy002.3ds',
    #                   r'T:\LT_data\TaAs\2016-01-04\Grid Spectroscopy002.3ds',
    #                   r'T:\LT_data\Copper\2019-12-22\Grid Spectroscopy002.3ds',
    #                   r'T:\LT_data\Copper\2014-03-20\Grid Spectroscopy002.3ds']
    kernel_size = (32, 32)  # Required argument to deconvolve both simulated and real measurements.

    # Option 1: Loading data from files
    # DS_handler = ThreeDSHandler(three_ds_paths[4], kernel_size)
    # real_measurement = DS_handler.get_measurement_data()

    # Option 2: Using a simulation
    measurement_size = (300, 300)
    energy_level_num = 2
    simulated = Simulation(energy_level_num, measurement_size, kernel_size,
                           SNR=20, defect_density=0.005)
    simulated.run()

    # Deconvolve:

    # deconv_real_result = deconvolve(real_measurement)
    deconv_sim_result = deconvolve(simulated.measurement)

    # Plotting
    plotting.slider_side_by_side(simulated.measurement.kernel,
                                 deconv_sim_result[0],
                                 simulated.measurement.density_of_states)


if __name__ == '__main__':
    #bench.test_benchmark()
     main()

## The commented code below was used to test functions as I wrote them ##
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
