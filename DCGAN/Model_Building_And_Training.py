# Importing dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Creating our Discriminator model
class Discriminator(nn.Module):
    def __init__(self, img_channels, features_d):
        super().__init__()
        # Input : N X channels_img X 64 X 64
        self.disc = nn.Sequential(nn.Conv2d(in_channels = img_channels,
                                            out_channels = features_d,
                                            kernel_size = 4,
                                            stride = 2,
                                            padding = 1),  # 32 X 32
                                  nn.LeakyReLU(negative_slope = 0.2),
                                  self._block(features_d, features_d *2, 4, 2, 1),  # 16 X 16
                                  self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 8 X 8
                                  self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 4 X 4
                                  nn.Conv2d(in_channels = features_d * 8,
                                            out_channels = 1,
                                            kernel_size = 4,
                                            stride = 2,
                                            padding = 0),   # 1 X 1
                                  nn.Sigmoid()
                                  )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = kernel_size,
                      stride = stride,
                      padding = padding,
                      bias = False),   # We are setting bias = False because we are using batch normalisation whose own bias term will cancel out the bias in this conv layer.
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope = 0.2),

        )
    def forward(self, x):
        return self.disc(x)

# Creating our generator model
class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g):
        super().__init__()
        # Input : N X z_dim X 1 X 1
        self.net = nn.Sequential(self._block(z_dim, features_g * 16, 4, 1, 0),   # N X features_g * 16 X 4 X 4
                                 self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8 X 8
                                 self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16 X 16
                                 self._block(features_g * 4, features_g * 4, 4, 2, 1),   # 32 X 32
                                 nn.ConvTranspose2d(in_channels = features_g * 4,
                                                    out_channels = img_channels,
                                                    kernel_size = 4,
                                                    stride = 2,
                                                    padding = 1),
                                 nn.Tanh()   # Using this activation function to get the output in the range in between range from -1 to 1.
                                 )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = kernel_size,
                               stride = stride,
                               padding = padding,
                               bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    """ Function to check whether the models are giving outputs of desired shapes. """
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print('Success')

test()

# Setting up the Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 2e-4
batch_size = 128
img_size = 64
img_channels = 1
z_dim = 100
num_epochs = 20
features_disc = 64
features_gen = 64
loss_fn = nn.BCELoss()

transforms = transforms.Compose([transforms.Resize(img_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5 for _ in range(img_channels)], [0.5 for _ in range(img_channels)])
                                                      ])
dataset = datasets.MNIST(root = "dataset/", train = True, transform = transforms, download = True)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
gen = Generator(z_dim = z_dim, img_channels = img_channels, features_g = features_gen).to(device)
disc = Discriminator(img_channels, features_d = features_disc).to(device)

# Initializing the weights for our models
initialize_weights(gen)
initialize_weights(disc)

# Setting up the optimizers
opt_gen = optim.Adam(gen.parameters(), lr = lr, betas = (0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr = lr, betas = (0.5, 0.999))

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        fake = gen(noise)

        # Training the discriminator
        disc_real = disc(real).reshape(-1)
        loss_disc_real = loss_fn(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph = True)
        opt_disc.step()

        # Training the generator
        output = disc(fake).reshape(-1)
        loss_gen = loss_fn(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Printing to tensorboard
        with torch.inference_mode():
            fake = gen(fixed_noise)
            img_grid_real = torchvision.utils.make_grid(
                real[:32], normalize = True
            )
            img_grid_fake = torchvision.utils.make_grid(fake[:32,],
                                                        normalize = True)
            writer_real.add_image('Real', img_grid_real, global_step = step)
            writer_fake.add_image('Fake', img_grid_fake, global_step = step)

        step += 1






