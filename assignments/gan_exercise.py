"""Exercise 4: GAN on MNIST (based on realpython.com/generative-adversarial-networks).

Structural bug: the last batch of an epoch is smaller than `batch_size`
(60000 is not divisible by 64), but the label tensors were built with a fixed
`batch_size`, so BCELoss sees mismatched sizes
(e.g. target [128, 1] vs input [96, 1]).
FIX: build labels and noise from the actual batch size (`real_samples.size(0)`).

Cosmetic bug: `if n == batch_size - 1` ties the progress image to the batch
size (an arbitrary batch index). It should show once per epoch, on the last
batch, i.e. `n == len(train_loader) - 1`.
Also saves figures to disk instead of using IPython `display`.
"""
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class Generator(nn.Module):
    """Maps 100-d noise to a 1x28x28 image."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 784), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).view(x.size(0), 1, 28, 28)


class Discriminator(nn.Module):
    """Outputs the probability that a 28x28 image is real."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(x.size(0), 784))


def _save_grid(images: torch.Tensor, title: str, path: str) -> None:
    fig = plt.figure()
    for i in range(16):
        sub = fig.add_subplot(4, 4, 1 + i)
        sub.imshow(images[i].reshape(28, 28), cmap="gray_r")
        sub.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train_gan(
    batch_size: int = 32,
    num_epochs: int = 100,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    out_dir: str = "gan_output",
) -> None:
    """Train the GAN and save a sample grid after every epoch to `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = torchvision.datasets.MNIST(
        root=".", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    loss_function = nn.BCELoss()
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            current = real_samples.size(0)  # FIX: last batch can be smaller
            real_samples = real_samples.to(device)
            real_labels = torch.ones((current, 1), device=device)
            fake_labels = torch.zeros((current, 1), device=device)

            # train discriminator on real + generated
            noise = torch.randn((current, 100), device=device)
            generated = generator(noise)
            all_samples = torch.cat((real_samples, generated))
            all_labels = torch.cat((real_labels, fake_labels))
            discriminator.zero_grad()
            loss_d = loss_function(discriminator(all_samples), all_labels)
            loss_d.backward()
            optimizer_d.step()

            # train generator to fool the discriminator
            noise = torch.randn((current, 100), device=device)
            generator.zero_grad()
            generated = generator(noise)
            loss_g = loss_function(discriminator(generated), real_labels)
            loss_g.backward()
            optimizer_g.step()

            if n == len(train_loader) - 1:  # FIX: once per epoch
                title = (f"Epoch: {epoch} Loss D.: {loss_d:.2f} "
                         f"Loss G.: {loss_g:.2f}")
                _save_grid(generated.detach().cpu().numpy(), title,
                           os.path.join(out_dir, f"epoch_{epoch:03d}.png"))


if __name__ == "__main__":
    train_gan(batch_size=64, num_epochs=50)
