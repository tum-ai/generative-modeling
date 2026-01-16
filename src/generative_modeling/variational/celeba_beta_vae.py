import torch
from torch import nn

class CelebABetaVAE(nn.Module):
    def __init__(self, latent_dim=128, image_size=150, hidden_dims=None):
        super(CelebABetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims
        self.final_dim = hidden_dims[-1]
        
        in_channels = 3
        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
        # Calculate size after encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size)
            dummy_output = self.encoder(dummy_input)
            self.encoder_output_shape = dummy_output.shape[1:]
            self.flattened_size = hidden_dims[-1] * self.encoder_output_shape[1] * self.encoder_output_shape[2]
            self.spatial_size = self.encoder_output_shape[1]

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)
        
        reversed_hidden_dims = list(reversed(hidden_dims))

        for i in range(len(reversed_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(reversed_hidden_dims[i],
                                       reversed_hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(reversed_hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(reversed_hidden_dims[-1],
                               reversed_hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(reversed_hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(reversed_hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())

    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.final_dim, self.spatial_size, self.spatial_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
