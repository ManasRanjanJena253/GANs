# Importing dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# Creating our discriminator model
class Discriminator(nn.Module):   # This model checks whether the image created our generator be considered an image of the given class or not.
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(nn.Linear(in_features = in_features,
                                            out_features = 128),
                                  nn.LeakyReLU(),
                                  nn.Linear(in_features = 128,
                                            out_features = 1),  # Out features is only 1 as we want to check if the image is real or not yes/no.
                                  nn.Sigmoid()) # Sigmoid function ensures that our output is in between 0 and 1
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):  # Here z_dim is the dimension of the latent noise that our generator might create.
        super().__init__()
        self.gen = nn.Sequential(nn.Linear(in_features = z_dim,
                                           out_features = 256),
                                 nn.LeakyReLU(),
                                 nn.Linear(in_features = 256,
                                           out_features = img_dim),
                                 nn.Tanh())
    def forward(self, x):
        return self.gen(x)

# Setting the devices and hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 3e-4      # Suggested by andrej kapathy to be the best learning rate for adam optimizer.
z_dim = 64
img_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50

disc = Discriminator(in_features = img_dim).to(device)
gen = Generator(z_dim = z_dim, img_dim = img_dim)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean = (0.5,), std = (0.5,)),
                                 transforms.TrivialAugmentWide(num_magnitude_bins = 20)
                                 ])

dataset = datasets.MNIST(root = 'dataset/', transform = transforms, download = True)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
opt_disc = optim.Adam(disc.parameters(), lr = lr)  # Optimizer for our discriminator
opt_gen = optim.Adam(disc.parameters(), lr = lr)   # Optimizer for our generator
loss_fn = nn.BCELoss()
step = 0

for epoch in tqdm(range(num_epochs)):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Train the discriminator: max og(D(real)) + log(1 - D(G(z))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = loss_fn(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2

        disc.zero_grad()
        lossD.backward(retain_graph = True)
        opt_disc.step()

        # Train the generator
        output = disc(fake).view(-1)
        lossG = loss_fn(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

