"""Render the Option 1 + LPIPS cVAE forward-pass graph to PDF via torchviz.

Architecture mirrors cVAE_train_option1_lpips.py exactly. Standalone — does not
import from cVAE_train.py so it can run on any box without the data pipeline.
"""

import os
import torch
import torch.nn as nn
from torchviz import make_dot


N_ATTRS      = 40
LATENT_DIM   = 256
ATTR_EMB_DIM = 128
IMAGE_SIZE   = 128


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attr_embed = nn.Sequential(
            nn.Linear(N_ATTRS, ATTR_EMB_DIM),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.convs = nn.Sequential(
            nn.Conv2d(3,   64,  4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        flat_dim = 512 * 4 * 4
        self.fc_mu     = nn.Linear(flat_dim + ATTR_EMB_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(flat_dim + ATTR_EMB_DIM, LATENT_DIM)

    def forward(self, x, attrs):
        h = self.convs(x).flatten(1)
        a = self.attr_embed(attrs)
        h = torch.cat([h, a], dim=1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attr_embed = nn.Sequential(
            nn.Linear(N_ATTRS, ATTR_EMB_DIM),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM + ATTR_EMB_DIM, 512 * 4 * 4),
        )
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64,  3,   4, stride=2, padding=1),
        )

    def forward(self, z, attrs):
        a = self.attr_embed(attrs)
        h = torch.cat([z, a], dim=1)
        h = self.fc(h).view(-1, 512, 4, 4)
        return torch.tanh(self.deconvs(h))


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, attrs):
        mu, logvar = self.encoder(x, attrs)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, attrs)
        return recon, mu, logvar


def main():
    here    = os.path.dirname(os.path.abspath(__file__))
    project = os.path.abspath(os.path.join(here, ".."))
    out_dir = os.path.join(project, "Generated")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cvae_graph")  # render() appends .pdf

    model = CVAE().eval()
    x     = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    attrs = torch.randn(1, N_ATTRS)

    recon, mu, logvar = model(x, attrs)
    dot = make_dot(
        (recon, mu, logvar),
        params=dict(model.named_parameters()),
        show_attrs=False,
        show_saved=False,
    )
    dot.format = "pdf"
    dot.render(out_path, cleanup=True)
    print(f"Saved: {out_path}.pdf")


if __name__ == "__main__":
    main()
