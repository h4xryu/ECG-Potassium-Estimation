import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np



# Define the beta-VAE model
class BetaVAE(nn.Module):
    def __init__(self, input_channels=12, seq_length=600, latent_dim=32, beta=1.0):
        super(BetaVAE, self).__init__()
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=5, dilation=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, kernel_size=5, dilation=2, padding=4),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, kernel_size=5, dilation=4, padding=8),
            nn.LeakyReLU(),
            nn.Conv1d(128, 64, kernel_size=5, dilation=8, padding=16),
            nn.AdaptiveMaxPool1d(1)
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 128, kernel_size=5, dilation=8, padding=16),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=5, dilation=4, padding=8),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=5, dilation=2, padding=4),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, input_channels, kernel_size=5, dilation=1, padding=2)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.fc_dec(z).unsqueeze(-1)
        x_recon = self.decoder(z)
        return x_recon

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss_function(self, x, x_recon, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + self.beta * kld_loss

# Training loop
def train_vae(model, dataloader, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            batch = batch.permute(0, 2, 1)  # (B, C, L)
            x_recon, mu, logvar = model(batch)
            loss = model.loss_function(batch, x_recon, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

# Example usage
# if __name__ == "__main__":
#     # Simulate ECG data (12 leads, 600 samples each)
#     num_samples = 1000
#     ecg_data = np.random.rand(num_samples, 12, 600)  # Replace with real ECG data
#
#     # Dataset and DataLoader
#     dataset = ECGDataset(ecg_data)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
#     # Initialize model, optimizer
#     model = BetaVAE(input_channels=12, seq_length=600, latent_dim=32, beta=4.0)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     # Train the model
#     train_vae(model, dataloader, optimizer, epochs=20)
