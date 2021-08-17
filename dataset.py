import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import io


class QPIDataSet(Dataset):
    def __init__(self, path2dataset):

        self.path2dataset = path2dataset
        self.files_in_folder = os.listdir(path2dataset)
        self.length = len(self.files_in_folder) // 3
        self.files_in_folder.sort()

        self.activation_map = self.files_in_folder[:self.length]
        self.kernel = self.files_in_folder[self.length:2 * self.length]
        self.measurement = self.files_in_folder[2 * self.length:3 * self.length]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        measurement = np.load(self.path2dataset + '/' + self.measurement[idx])
        kernel = np.load(self.path2dataset + '/' + self.kernel[idx])
        activation = io.mmread(self.path2dataset + '/' + self.activation_map[idx]).tolil()

        measurement = torch.FloatTensor(measurement)
        kernel = torch.FloatTensor(kernel)
        activation = torch.FloatTensor(activation.A)

        if torch.cuda.is_available():
            return measurement.cuda(), kernel.cuda(), activation.cuda()
        return measurement, kernel, activation
