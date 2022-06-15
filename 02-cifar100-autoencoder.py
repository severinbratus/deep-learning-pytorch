import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from os import path
import matplotlib.pyplot as plt
import numpy as np


def prepare_data(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR100(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR100(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def interpolate_images(images, model, device, steps=5):
    model.eval()
    with torch.no_grad():
        latents = [model.encoder(img.unsqueeze(0).to(device)) for img in images]
        grid = []
        for i in range(steps):
            row = []
            for j in range(steps):
                x = i / (steps - 1)
                y = j / (steps - 1)
                latent_interp = ((1 - x) * (1 - y) * latents[0] + x * (1 - y) * latents[1]
                               + (1 - x) * y * latents[2] + x * y * latents[3])
                output = model.decoder(latent_interp)
                row.append(output.squeeze(0).cpu().numpy().transpose(1, 2, 0))
            grid.append(row)
        return np.array(grid)


def display_interpolation(grid):
    steps = len(grid)
    fig, axes = plt.subplots(steps, steps, figsize=(12, 12))
    for i in range(steps):
        for j in range(steps):
            axes[i, j].imshow((grid[i][j] * 0.5 + 0.5).clip(0, 1))
            axes[i, j].axis('off')
    plt.tight_layout()
    # set_aspect('equal')
    plt.savefig("figs/interpolated.png")


def train(model, train_loader, test_loader, device, epochs=20, lr=1e-3):
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, _ in train_loader:
            images = images.to(device)

            opt.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, images)
            loss.backward()
            opt.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}]; Loss: {avg_train_loss:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_data()

    model = AutoEncoder().to(device)

    if path.exists("models/cifar-autoencoder.pt"):
        model.load_state_dict(torch.load("models/cifar-autoencoder.pt"))
    else:
        train(model, train_loader, test_loader, device, epochs=20, lr=1e-3)
        torch.save(model.state_dict(), "models/cifar-autoencoder.pt")

    # make interpolation grid

    data_iter = iter(train_loader)
    images, _ = next(data_iter)
    images4 = images[:4]

    grid = interpolate_images(images4, model, device)

    display_interpolation(grid)
