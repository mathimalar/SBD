import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


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


def smooth_abs(x, eps=1e-18):
    return (x ** 2 + eps) ** 0.5 - (eps ** 0.5)


def normalize_tensor_0to1(t):
    """
    Normalizes a tensor such that the data in each sample in the batch will distribute between 0 and 1.
    """
    shape = t.size()
    t = t.view(t.size(0), -1)
    t -= t.min(1, keepdim=True)[0]
    t /= t.max(1, keepdim=True)[0]
    return t.view(shape)


def normalize_tensor_sumto1(t):
    """
    Normalizes a tensor such that the data in each sample in the batch will sum to 1.
    """
    shape = t.size()
    t = t.view(t.size(0), -1)
    sum_tensor = t.sum(1, keepdim=True)
    sum_tensor[sum_tensor == 0] = 1
    t /= sum_tensor
    return t.view(shape)


class ActivationLoss(nn.Module):
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
        regulation_term = smooth_abs(self.regulator(activation_pred) - self.regulator(activation))
        loss = F.huber_loss(conv_pred_stack, conv_target_stack) / 2
        loss += self.r * regulation_term
        # loss += (self.r * 10 ** - 4) * torch.abs(torch.count_nonzero(activation) - torch.count_nonzero(activation_pred)) / torch.count_nonzero(activation)
        loss /= activation.shape[0]
        if torch.cuda.is_available():
            return loss.cuda()
        return loss


class KernelLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, kernel_pred, activation, kernel, target):
        conv_pred = []
        conv_target = []
        for i in range(kernel_pred.shape[0]):
            single_kernel = kernel[i].unsqueeze(dim=0)
            single_kernel_pred = kernel_pred[i].unsqueeze(dim=0)
            single_activation = activation[i].unsqueeze(dim=0).unsqueeze(dim=0)
            # plot_conv(single_kernel, single_activation, target[i])
            conv_pred.append(F.conv2d(single_activation, single_kernel_pred, padding='same'))
            conv_target.append(F.conv2d(single_activation, single_kernel, padding='same'))
        conv_pred_stack = torch.stack(conv_pred, dim=0).squeeze(dim=1)
        conv_target_stack = torch.stack(conv_target, dim=0).squeeze(dim=1)
        loss = F.huber_loss(conv_pred_stack, conv_target_stack) / 2
        loss /= activation.shape[0]
        if torch.cuda.is_available():
            return loss.cuda()
        return loss
