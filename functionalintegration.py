import PIL
import numpy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import imageio
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from decoder import Decoder

batch_size = 512
epochs = 2
sample_size = 10 # fixed sample size
nz = 1 # size of generator input
k = 1 # number of steps to apply to the discriminator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)),])

train_data = datasets.MNIST(
    root='../input/data',
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)



class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

#generator = Generator(nz).to(device)

num_encoder_layers: int = 6
num_decoder_layers: int = 6
dim_model: int = 512
num_heads: int = 6
dim_feedforward: int = 2048
dropout: float = 0.1
cross_dimensions = 10
decoder = Decoder(number_of_blocks=num_decoder_layers,
            d_model=dim_model,
            num_heads=num_heads,
            d_ff=dim_feedforward,
            dropout=dropout,
            cross_dimensions=cross_dimensions)

generator = torch.nn.Transformer(d_model = 512, nhead = 8, num_encoder_layers = 6, num_decoder_layers = 6,
                                   dim_feedforward = 2048, dropout = 0.1,
                                   custom_decoder = decoder)


optim_g = optim.Adam(generator.parameters(), lr=0.0002)

images = []



def create_random_seed(sample_size, nz):
    return torch.randn(sample_size, nz).to(device)


for epoch in range(epochs):
    print("Epoch " + str(epoch) + " of " + str(epochs))
    loss_g = 0.0

    #image_seed = create_random_seed(sample_size, nz)
    #generated_img = generator(image_seed).cpu().detach()

    image_seed = torch.randn(sample_size, nz, 512).to(device)
    classVector = torch.rand(cross_dimensions, 1)
    generated_img = generator(image_seed, image_seed, tgt_mask=classVector).cpu().detach()
    images.append(generated_img)

imgs = images[-1]
imgs = imgs.view(-1, 1, 28, 28)
imgs = torch.squeeze(imgs)
imgs = imgs*255
imgs = numpy.array(imgs, dtype=numpy.uint8)
imageio.mimsave('generator_images.gif', imgs)

#def convert_image_np(inp):
    #"""Convert a Tensor to numpy image."""
    #inp = inp * 255
    #inp = numpy.array(inp, dtype=numpy.uint8)
#    return PIL.Image.fromarray(inp)

#test = convert_image_np(imgs)

plt.imshow(imgs)
plt.show()

