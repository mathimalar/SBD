from matplotlib import pyplot as plt
import numpy as np
import SBD

# Testing max_submatrix_pos

m1, m2 = 2, 2
sample = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])  # np.random.randint(0, 100, size=(5, 30, 30))
i, j = SBD.max_submatrix_pos(sample, m1, m2)
flat_sample = np.sum(sample, axis=0)
# plt.imshow(np.sum(sample, axis=0), interpolation='nearest')
# plt.show()
# print(flat_sample)
# print(flat_sample[i: i + m2, j: j + m2])
# print(f'i = {i}, j={j}')

# Testing crop_to_center

kernel_shape = np.array([3, 6, 6])
a = np.random.randint(0, 10, size=kernel_shape)
padding = np.floor(kernel_shape[1:] / 2).astype('int')
padded = np.pad(a, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])))
# print(f'padded = \n{padded}\ncropped = \n{SBD.crop_to_center(padded, kernel_shape)}')

# Testing shift

arr = np.ones([3, 7, 7])
arr[:, 3:6, 1:4] = 20
c_arr, c_x = SBD.center((2, 2), arr, arr)
# print(f'a =\n{arr}\n shifted =\n{c_arr}\n anti-shifted =\n{c_x}')

# testing kernel_factory

# sample_ker = SBD.kernel_factory(levels, 100, 100)
# fig, ax = plt.subplots(1, levels)
# fig.suptitle('Horizontally stacked subplots')
# for i in range(levels):
#     ax[i].imshow(sample_ker[i, :, :], cmap='hot', interpolation='nearest')
# plt.show()

# testing Y_factory
levels = 2
density = 0.005
SNR = 2
Y, A, X = SBD.Y_factory(levels, (185, 185), (25, 25), density, SNR)

fig, ax = plt.subplots(levels, 2)
fig.suptitle('Y:')
for i in range(levels):
    ax[i, 0].imshow(A[i, :, :], cmap='hot', interpolation='nearest')
    ax[i, 1].imshow(Y[i, :, :], cmap='hot', interpolation='nearest')
plt.show()

X_new, total_iter, cost = SBD.FISTA(0.001, A, Y)

print(np.sum(X - X_new))

