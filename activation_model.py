import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from model_tools import ActivationLoss
import model_tools
from dataset import QPIDataSet
from matplotlib import pyplot as plt
import numpy as np

loss_func = ActivationLoss(0.01)


def plot_example(measurement, activation, pred_activation, idx):
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
    fig.suptitle(f'Prediction')
    ax[0].set_title('Input', fontsize=12)
    ax[0].imshow(measurement[idx][0].cpu(), cmap='hot')

    ax[1].set_title('Network output', fontsize=12)
    ax[1].imshow(pred_activation[idx][0].cpu(), cmap='hot')

    ax[2].set_title('Target', fontsize=12)
    ax[2].imshow(activation[idx].cpu(), cmap='hot')

    for i in range(3):
        ax[i].set_axis_off()
    plt.savefig(f'C:/Users/physicsuser/Documents/SBD/pl_predictions/Pred{idx}')
    pass


class QPIDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.train_dims = None

    def prepare_data(self):
        # called only on 1 GPU
        measurement_shape = (1, 128, 128)
        kernel_shape = (16, 16)

        model_tools.save_data(20000, measurement_shape, kernel_shape, training=True)
        model_tools.save_data(1000, measurement_shape, kernel_shape, validation=True)
        model_tools.save_data(500, measurement_shape, kernel_shape, testing=True)

    def setup(self, stage=None):
        # called on every GPU
        self.train = QPIDataSet(os.getcwd() + '/training_dataset')
        self.val = QPIDataSet(os.getcwd() + '/validation_dataset')
        self.test = QPIDataSet(os.getcwd() + '/testing_dataset')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=0)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=0)


class ActivationSys(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, input_channels=1):
        super().__init__()
        self.learning_rate = learning_rate
        self.input_channels = input_channels
        pad = 2
        ker_same = (5, 5)  # These values make sure the image dimensions stay the same
        self.down1 = Down(2 ** 2, 2 ** 4)
        self.down2 = Down(2 ** 4, 2 ** 6)
        self.up1 = Up(2 ** 6, 2 ** 4)
        self.up2 = Up(2 ** 4, 2 ** 2)
        self.features = nn.Sequential(

            # Input layer
            nn.Conv2d(input_channels, 2 ** 2, kernel_size=ker_same, padding=pad, bias=False),
            nn.BatchNorm2d(2 ** 2),
            nn.LeakyReLU(),

            # Hidden layers:

            self.down1,

            self.down2,

            self.up1,

            self.up2,

            # Output layer

            nn.Conv2d(2 ** 2, 1, kernel_size=ker_same, padding=pad, bias=False),
            nn.ReLU()
        )

    def forward(self, x_in):
        x = x_in / torch.linalg.norm(x_in)
        x = self.features(x)
        x = x / torch.sum(x) if torch.sum(x) != 0 else x
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        measurement, kernel, activation = batch
        pred_activation = self(measurement)
        loss = loss_func(pred_activation, activation, kernel, measurement)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        measurement, kernel, activation = batch
        pred_activation = self(measurement)
        loss = loss_func(pred_activation, activation, kernel, measurement)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        measurement, kernel, activation = batch
        pred_activation = self(measurement)
        loss = loss_func(pred_activation, activation, kernel, measurement)
        self.log("valid_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        measurement, kernel, activation = batch
        pred_activation = self(measurement)
        if batch_idx > 0:
            return pred_activation
        idxs = np.random.choice(measurement.shape[0], 10, replace=False)
        for idx in idxs:
            plot_example(measurement, activation, pred_activation, idx)
        return pred_activation

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) *
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)
