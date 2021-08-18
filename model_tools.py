import numpy as np
import os
from SBD import Y_factory
from scipy import io
import torch.nn as nn
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


def save_data(number_of_samples, measurement_size, kernel_size, SNR=2, training=False, validation=False, testing=False):
    files_in_folder = os.listdir()
    if training and 'training_dataset' not in files_in_folder:
        os.system("mkdir training_dataset")
    if validation and 'validation_dataset' not in files_in_folder:
        os.system("mkdir validation_dataset")
    if testing and 'testing_dataset' not in files_in_folder:
        os.system("mkdir testing_dataset")

    defect_density = np.random.uniform(low=-4, high=-2, size=(number_of_samples,))

    E, n1, n2 = measurement_size
    for i in range(number_of_samples):
        temp_measurement, temp_kernel, temp_activation_map = Y_factory(E, (n1, n2),
                                                                       kernel_size,
                                                                       10 ** defect_density[i],
                                                                       SNR)
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


measurement_shape = (1, 200, 200)
kernel_shape = (25, 25)

save_data(10000, measurement_shape, kernel_shape, training=True)
save_data(1000, measurement_shape, kernel_shape, validation=True)
save_data(100, measurement_shape, kernel_shape, testing=True)


def tensor_to_nparray(tensor):
    return np.array(tensor.squeeze().cpu().detach().numpy())


def plot_conv(kernel_in, activation_in, target_in):
    A_conv_K = tensor_to_nparray(F.conv2d(activation_in, kernel_in, padding='same'))
    activation = tensor_to_nparray(activation_in)
    kernel = tensor_to_nparray(kernel_in)
    target = tensor_to_nparray(target_in)

    fig, ax = plt.subplots(1, 4, figsize=(9, 3), dpi=150)
    ax[0].set_title('Kernel', fontsize=12)
    ax[0].imshow(kernel, cmap='hot')

    ax[1].set_title('Activation Pred', fontsize=12)
    ax[1].imshow(activation, cmap='hot')

    ax[2].set_title('Convolution', fontsize=12)
    ax[2].imshow(A_conv_K, cmap='hot')

    ax[3].set_title('Measurement', fontsize=12)
    ax[3].imshow(target, cmap='hot')

    for i in range(4):
        ax[i].set_axis_off()
    plt.show()
    pass


def smooth_abs(x, k=100):
    return (2/k) * torch.log(1 + torch.exp(k*x)) - x - (2/k) * np.log(2)


class RegulatedLoss(nn.Module):
    def __init__(self, r=0.1):
        super().__init__()
        self.r = r
        self.mu = 10 ** -6

    def regulator(self, activation):
        return torch.sum(self.mu ** 2 * (torch.sqrt(1 + (self.mu ** -2) * torch.abs(activation)) - 1))

    def forward(self, activation_pred, activation, kernel, target):
        conv_pred = []
        conv_target = []
        for i in range(activation_pred.shape[0]):
            single_kernel = kernel[i].unsqueeze(dim=0)
            single_activation_pred = activation_pred[i].unsqueeze(dim=0)
            single_activation = activation[i].unsqueeze(dim=0).unsqueeze(dim=0)
            # plot_conv(single_kernel, single_activation, target[i])
            conv_pred.append(F.conv2d(single_activation_pred, single_kernel, padding='same'))
            conv_target.append(F.conv2d(single_activation, single_kernel, padding='same'))
        conv_pred_stack = torch.stack(conv_pred, dim=0).squeeze(dim=1)
        conv_target_stack = torch.stack(conv_target, dim=0).squeeze(dim=1)
        regulation_term = self.regulator(activation_pred) - self.regulator(activation)
        loss = F.huber_loss(conv_pred_stack, conv_target_stack) / 2 \
               + self.r * smooth_abs(regulation_term)
        loss /= activation.shape[0]
        if torch.cuda.is_available():
            return loss.cuda()
        return loss
