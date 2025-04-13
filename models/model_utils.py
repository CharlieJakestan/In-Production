# models/model_utils.py

import torch
import torch.nn as nn

class MakeupApplicationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A very simple autoencoder-like structure (starter template)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def train_step(self, input_image, target_image, optimizer):
        self.train()
        optimizer.zero_grad()
        output = self(input_image)
        loss = nn.functional.l1_loss(output, target_image)
        loss.backward()
        optimizer.step()
        return loss.item()

