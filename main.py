from matplotlib import pyplot as plt
import numpy as np
import SBD
import pylops as pl
import torch.nn.functional as F


def side_by_side(M1, M2, figtitle, ax1_title, ax2_title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(figtitle)
    ax[0].imshow(M1, cmap='hot')
    ax[0].set_title(ax1_title)
    ax[1].imshow(M2, cmap='hot')
    ax[1].set_title(ax2_title)
    for i in range(2):
        ax[i].set_axis_off()
    plt.tight_layout()
    plt.show()


levels = 1
density = 0.005
SNR = 10
n1, n2 = 512, 512
m1, m2 = 32, 32

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

# Testing RTRM & measurement_to_activation & recovery_error & compare_fft_plot:
# It all works (compare fft shows checkers-board pattern).
# Y, A, X = SBD.Y_factory(levels, (n1, n2), (m1, m2), density, SNR)
# side_by_side(A[0], Y[0], 'Measurement and kernel', 'Kernel', 'Measurement')
#
# # A_noise = A + np.random.normal(0, A.mean() / 2, (levels, m1, m2))
# # A_noise = SBD.sphere_norm_by_layer(A_noise)
#
# A_rand = np.random.normal(0, 1, (levels, 2*m1, 2*m2))
# A_rand[0] = A_rand[0] / np.linalg.norm(A_rand[0])
#
#
#
# X_guess_cnn = SBD.measurement_to_activation(Y, model='cnn')
# X_guess_lista = SBD.measurement_to_activation(Y, model='lista')
# side_by_side(X_guess_cnn, X.A, 'CNN Guess', 'Predicted', 'True')
# side_by_side(X_guess_cnn, Y[0], 'CNN Pred vs Measurement', 'Predicted Activation', 'Measurement')
#
# side_by_side(X_guess_lista, X.A, 'LISTA Guess', 'Predicted', 'True')
# side_by_side(X_guess_lista, Y[0], 'LISTA Pred vs Measurement', 'Predicted kernel', 'Measurement')
#
# A_solved_cnn = SBD.RTRM(1e-5, X_guess_cnn, Y, A_rand)
# A_solved_cnn = SBD.crop_to_center(A_solved_cnn, (levels, m1, m2))
# A_solved_cnn = A_solved_cnn / np.linalg.norm(A_solved_cnn)  # norm = 1
#
# A_solved_lista = SBD.RTRM(1e-5, X_guess_lista, Y, A_rand)
# A_solved_lista = SBD.crop_to_center(A_solved_lista, (levels, m1, m2))
# A_solved_lista = A_solved_lista / np.linalg.norm(A_solved_lista)
#
# count0 = np.count_nonzero(X.A)
# count_cnn = np.count_nonzero(X_guess_cnn)
# count_lista = np.count_nonzero(X_guess_cnn)
#
# cnn_error = SBD.recovery_error(A_solved_cnn[0], A[0])
# lista_error = SBD.recovery_error(A_solved_lista[0], A[0])
# side_by_side(A_solved_cnn[0], A[0], f'CNN Kernels. Error = {np.round(cnn_error, 4)}', 'Predicted', 'True')
# side_by_side(A_solved_lista[0], A[0], f'LISTA Kernels. Error = {np.round(lista_error, 4)}', 'Predicted', 'True')
# SBD.compare_fft_plot(A[0], A_solved_lista[0], 'True FFT vs SBD FFT')

# Testing benchmark:

extent = [10**-3, 10**-1, 8/256, 62/256]
# defect_range = np.logspace(-3, -1, 5)
# kernel_range = np.linspace(8, 62, 5, dtype=int)
#
# error_matrix = SBD.benchmark('lista', defect_range, kernel_range, samples=4)
# np.save('benchmark', error_matrix)

error_matrix = np.load('benchmark.npy')
plt.imshow(error_matrix, origin='lower', cmap='jet', extent=extent, aspect='auto')
plt.colorbar(label=r'$\epsilon$')
plt.xlabel('Defect density')
plt.ylabel(r'Kernel relative size $m/n$')
plt.xscale('log')
plt.clim(0, 1)
plt.show()
