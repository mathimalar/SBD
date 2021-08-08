from matplotlib import pyplot as plt
import numpy as np
import SBD


def side_by_side(M1, M2, title):
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(title)
    ax[0].imshow(M1)
    ax[1].imshow(M2)
    plt.show()


levels = 3
density = 0.005
SNR = 2
n1, n2 = 185, 185
m1, m2 = 25, 25

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
Y, A, X = SBD.Y_factory(levels, (n1, n2), (m1, m2), density, SNR)
A_noise = A + np.random.normal(0, A.mean() / 2, (levels, m1, m2))
A_noise = SBD.sphere_norm_by_layer(A_noise)
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
#     plt.colorbar()
#
# fig2, ax2 = plt.subplots()
# ax2.imshow(X.toarray(), cmap='hot', interpolation='nearest')
# ax2.set_xticks([])
# ax2.set_yticks([])
# fig2.savefig('X.jpg', dpi=500)
# plt.show()

# Testing cost function - WORKS
cost = SBD.cost_fun(0.05, A, X, Y)
print(cost)
cost = SBD.cost_fun(0.05, A_noise, X, Y)
print(cost)

# Testing FISTA: - gives bad approximations for activation maps
# levels = 3
# # density = 0.005
# # SNR = 2
# # n1, n2 = 185, 185
# # m1, m2 = 25, 25
# # Y, A, X = SBD.Y_factory(levels, (n1, n2), (m1, m2), density, SNR)
# # A_noise = A + np.random.normal(0, A.mean() / 2, (levels, m1, m2))
# # A_noise = SBD.sphere_norm_by_layer(A_noise)

X_dense = X.toarray()
X_new = SBD.FISTA(1e-6, A, Y, niter=50)

fig, ax = plt.subplots(1, 2)
fig.suptitle('Activation maps')

im = ax[0].imshow(X_new)
ax[1].imshow(X_dense)
fig.colorbar(im, ax=ax[0])

plt.show()

error_X = np.sum(X - X_new) / np.sum(X)
print(f'X relative error: {round(100 * error_X)}%')


# Testing RTRM:
# A_rand = np.random.normal(0, A.mean() / 2, (levels, m1, m2))
# A_solved = SBD.RTRM(1e-5, X, Y, A_rand)
# side_by_side(A_solved[0], A[0], 'RTRM result')

