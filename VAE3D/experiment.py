import os
import torch
import torchvision.utils as vutils
from torch import optim
from models import BaseVAE
from models.types_ import *

class VAEXperiment:

    def __init__(self, vae_model: BaseVAE, params: dict, datamodule):
        self.model = vae_model
        self.params = params
        self.datamodule = datamodule
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.current_epoch = 0
        self.optimizer = self.configure_optimizers()

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def train_step(self, batch):
        #print(batch.shape)
        real_img = batch
        labels = batch
        real_img, labels = real_img.to(self.device), labels.to(self.device)

        results = self.forward(real_img)
        train_loss = self.model.loss_function(*results, M_N=self.params['kld_weight'])

        self.optimizer.zero_grad()
        train_loss['loss'].backward()
        self.optimizer.step()

        return train_loss

    def val_step(self, batch):
        real_img = batch
        labels = batch
        real_img, labels = real_img.to(self.device), labels.to(self.device)

        with torch.no_grad():
            results = self.forward(real_img, labels=labels)
            val_loss = self.model.loss_function(*results, M_N=1.0)

        return val_loss

    def sample_images(self):
        test_input = next(iter(self.datamodule.test_dataloader()))
        test_label = next(iter(self.datamodule.test_dataloader()))
        test_input, test_label = test_input.to(self.device), test_label.to(self.device)

        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          os.path.join("logs",
                                       "Reconstructions",
                                       f"recons_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        return optimizer

    def run_training_epoch(self, train_loader):
        self.model.train()
        for batch in train_loader:
            train_loss = self.train_step(batch)

    def run_validation_epoch(self, val_loader):
        self.model.eval()
        for batch in val_loader:
            val_loss = self.val_step(batch)

    def fit(self, epochs, train_loader, val_loader):
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.run_training_epoch(train_loader)
            self.run_validation_epoch(val_loader)
            self.sample_images()