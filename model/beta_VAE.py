import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 in_channels=12,
                 channels=256,
                 depth=5,
                 reduced_size=128,
                 out_channels=64,
                 kernel_size=5,
                 dropout=0.3,
                 softplus_eps=1.0e-4,
                 sd_output=True):
        super().__init__()

        self.sd_output = sd_output
        self.softplus_eps = softplus_eps

        # Create list of convolutional layers with specified depth
        conv_layers = []
        current_channels = in_channels

        for i in range(depth):
            conv_layers.extend([
                nn.Conv1d(
                    current_channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            current_channels = channels

        # Final reduction layer
        conv_layers.append(
            nn.Conv1d(
                channels,
                out_channels,
                kernel_size=1
            )
        )

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate conv output size and create dense layers
        self.fc = nn.Sequential(
            nn.Linear(out_channels, channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )

        # Output layers
        self.fc_mu = nn.Linear(channels, reduced_size)
        if sd_output:
            self.fc_logvar = nn.Linear(channels, reduced_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Apply conv layers
        x = self.conv_layers(x)
        x = x.permute(0,2,1)
        # x = x.view(batch_size, -1)
        # x = x.permute(1,0)
        x = self.fc(x)

        # Get latent parameters
        mu = self.fc_mu(x)

        if self.sd_output:
            logvar = self.fc_logvar(x)
            # Use softplus for variance with epsilon for numerical stability
            std = F.softplus(logvar) + self.softplus_eps
            return mu, std
        return mu


class Decoder(nn.Module):
    def __init__(self,
                 k=2,
                 width=600,
                 in_channels=64,
                 channels=256,
                 depth=5,
                 out_channels=12,
                 kernel_size=5,
                 gaussian_out=True,
                 softplus_eps=1.0e-4,
                 dropout=0.0):
        super().__init__()

        self.gaussian_out = gaussian_out
        self.softplus_eps = softplus_eps

        # Initial projection
        self.fc = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )

        # Create transposed conv layers
        deconv_layers = []
        current_channels = channels

        for i in range(depth):
            deconv_layers.extend([
                nn.ConvTranspose1d(
                    current_channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            current_channels = channels

        # Output layer
        if gaussian_out:
            # For Gaussian output, we need both mean and standard deviation
            deconv_layers.append(
                nn.ConvTranspose1d(
                    channels,
                    out_channels * 2,  # Double channels for mean and std
                    kernel_size=1
                )
            )
        else:
            deconv_layers.append(
                nn.ConvTranspose1d(
                    channels,
                    out_channels,
                    kernel_size=1
                )
            )

        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.width = width
        self.k = k

    def forward(self, z):
        batch_size = z.size(0)

        # Initial projection and reshape
        x = self.fc(z)
        # x = x.view(batch_size, -1, self.k)
        x = x.permute(0,2,1)

        # Apply deconv layers
        x = self.deconv_layers(x)

        if self.gaussian_out:
            # Split channels into mean and log variance
            mean, logvar = torch.chunk(x, 2, dim=1)
            # Use softplus for variance with epsilon for numerical stability
            std = F.softplus(logvar) + self.softplus_eps
            return mean, std
        return x


class BetaVAE(nn.Module):
    def __init__(self,
                 in_channels=12,
                 channels=128,
                 depth=5,
                 latent_dim=64,
                 width=600,
                 beta=1.1,
                 kernel_size=5,
                 encoder_dropout=0.3,
                 decoder_dropout=0.0,
                 softplus_eps=1.0e-4):
        super().__init__()

        self.beta = beta

        # Initialize encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            channels=channels,
            depth=depth,
            reduced_size=64,
            out_channels=latent_dim,
            kernel_size=kernel_size,
            dropout=encoder_dropout,
            softplus_eps=softplus_eps,
            sd_output=True
        )

        # Initialize decoder
        self.decoder = Decoder(
            k=32,
            width=width,
            in_channels=latent_dim,
            channels=channels,
            depth=depth,
            out_channels=in_channels,
            kernel_size=kernel_size,
            gaussian_out=False,
            softplus_eps=softplus_eps,
            dropout=decoder_dropout
        )

    def reparameterize(self, mu, std):
        if self.training:
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # Encode
        mu, std = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, std)

        # Decode
        # recon_mu, recon_std = self.decoder(z)

        # return {
        #     'recon_mu': recon_mu,
        #     'recon_std': recon_std,
        #     'mu': mu,
        #     'std': std,
        #     'z': z
        # }
        return self.decoder(z), mu, std

    def compute_loss(self, x, output_dict, reduction='mean'):
        """
        Compute the ELBO loss for beta-VAE

        Args:
            x: Input tensor
            output_dict: Dictionary containing model outputs
            reduction: Loss reduction method ('mean' or 'sum')
        """
        recon_mu = output_dict['recon_mu']
        recon_std = output_dict['recon_std']
        mu = output_dict['mu']
        std = output_dict['std']

        # Reconstruction loss (negative log likelihood)
        # Using Gaussian likelihood
        recon_loss = 0.5 * torch.pow((x - recon_mu) / recon_std, 2) + torch.log(recon_std)

        if reduction == 'mean':
            recon_loss = recon_loss.mean()
        else:
            recon_loss = recon_loss.sum()

        # KL divergence
        # For Gaussian with diagonal covariance, KL has a closed form
        kl_div = -0.5 * torch.sum(1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2))

        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_div

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_div': kl_div
        }
