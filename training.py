import torch.jit
import matplotlib.pyplot as plt
from activation_model import ActivationSys, QPIDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar


qpi_dm = QPIDataModule()
model = ActivationSys()
model.load_state_dict(torch.load('trained_model_norm.pt'))
trainer = pl.Trainer(auto_lr_find=True, gpus=-1, auto_select_gpus=True,
                     callbacks=ProgressBar(), max_epochs=100)
# Run learning rate finder
lr_finder = trainer.tuner.lr_find(model, qpi_dm)

# Results can be found in
lr_finder.results

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()
# trainer.fit(model, qpi_dm)
# trainer.predict(model, qpi_dm)
