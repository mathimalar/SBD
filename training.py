import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model_tools import RegulatedLoss
from dataset import QPIDataSet
from model import KerNet, ActivationNet
from torch.optim import Adam
from tqdm import tqdm
import os
from pathlib import Path
from matplotlib import pyplot as plt


def plot_result(val_vs_epoch):
    plt.plot(val_vs_epoch)
    plt.ylabel('Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch number')
    plt.savefig('Validation_vs_loss')
    plt.show()


def plot_example(dataset, network, idx):
    meas, ker, act = dataset[idx]

    fig, ax = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
    fig.suptitle(f'Prediction {idx}')
    ax[0].set_title('Input', fontsize=12)
    ax[0].imshow(meas[0].cpu(), cmap='hot')

    ax[1].set_title('Network output', fontsize=12)
    network.eval()
    network.cpu()
    predicted = network(meas.unsqueeze(1).cpu())[0][0].data.numpy()
    ax[1].imshow(predicted, cmap='hot')

    ax[2].set_title('Target', fontsize=12)
    ax[2].imshow(act.cpu(), cmap='hot')

    for i in range(3):
        ax[i].set_axis_off()
    plt.savefig(f'Pred{idx}')


def compute_loss(dataloader, network, loss_function):
    loss = 0

    if torch.cuda.is_available():
        network.cuda()
    network.eval()

    n_batches = 0
    with torch.no_grad():
        for mes, ker, act in dataloader:
            n_batches += 1

            if torch.cuda.is_available():
                mes = mes.cuda()
                ker = ker.cuda()
                act = act.cuda()
            act_pred = network(mes)

            loss += loss_function(act_pred, act, ker, mes)

    loss = loss / n_batches
    return loss


print('Loading QPI datasets.')
batch_size = 10
train_ds = QPIDataSet(os.getcwd() + '/training_dataset')
valid_ds = QPIDataSet(os.getcwd() + '/validation_dataset')
training_dataloader = DataLoader(train_ds, batch_size=batch_size)
valid_dataloader = DataLoader(valid_ds, batch_size=batch_size)

# Defining the network, optimizer and loss function.
net = ActivationNet()
optimizer = Adam(net.parameters(), lr=1e-4)
loss_func = RegulatedLoss()

#  Getting parameters from my last model, and loading loss.
trained_model_path = Path('trained_model.pt', map_location=torch.device('cpu'))
loss_vs_epoch_path = Path('val_vs_loss.npy')
if loss_vs_epoch_path.is_file():
    validation_loss_vs_epoch = np.load(loss_vs_epoch_path).tolist()
    print('Loading previous loss.')
else:
    validation_loss_vs_epoch = []
if trained_model_path.is_file():
    net.load_state_dict(torch.load(trained_model_path))
    print('Loading parameters from your last model.')
    net.eval()

# Training loop:


n_epochs = 0


if torch.cuda.is_available():
    net.cuda()
    print('Using GPU.')

pbar = tqdm(range(n_epochs))

for epoch in pbar:

    if len(validation_loss_vs_epoch) > 1:
        rounded_loss = np.format_float_scientific(validation_loss_vs_epoch[-1], exp_digits=2, precision=4)
        rounded_best = np.format_float_scientific(np.min(validation_loss_vs_epoch), exp_digits=2, precision=4)
        pbar.set_description(f'epoch: {epoch}.'
                             f' val loss: {rounded_loss}. best: {rounded_best}')

    net.train()  # put the net into "training mode"
    for measurement, kernel, activation in training_dataloader:
        if torch.cuda.is_available():
            measurement = measurement.cuda()
            kernel = kernel.cuda()

        optimizer.zero_grad()
        pred_activation = net(measurement)
        loss = loss_func(pred_activation, activation, kernel, measurement)
        loss.backward()
        optimizer.step()

    net.eval()  # put the net into evaluation mode

    valid_loss = compute_loss(valid_dataloader, net, loss_func)

    validation_loss_vs_epoch.append(valid_loss.cpu().detach().numpy())

    if len(validation_loss_vs_epoch) == 1 or validation_loss_vs_epoch[-1] == min(validation_loss_vs_epoch):
        torch.save(net.state_dict(), 'trained_model.pt')
        np.save(loss_vs_epoch_path, validation_loss_vs_epoch)

# Plotting results
plot_result(validation_loss_vs_epoch)

# Plot one example for reference
idxs = np.random.choice(valid_ds.length, 10, replace=False)
for i in idxs:
    plot_example(valid_ds, net, i)
