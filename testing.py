import argparse
from typing import Union, Callable, Optional

import torch
import torchvision
from torch import optim

from decoder import Decoder, feed_forward






def position_encoding(
    seq_len: int, dim_model: int, device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim // dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    #softmax = torch.nn.functional.softmax(temp / scale)#, dim=-1)
    softmax = torch.softmax(temp / scale, dim=-1)
    return torch.bmm(softmax,value)#softmax.bmm(value)
class AttentionHead(torch.nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = torch.nn.Linear(dim_in, dim_q)
        self.k = torch.nn.Linear(dim_in, dim_k)
        self.v = torch.nn.Linear(dim_in, dim_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = torch.nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )
class Residual(torch.nn.Module):
    def __init__(self, sublayer: torch.nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = torch.nn.LayerNorm(dimension)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))
class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)


class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src = src + position_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)

        return src
class Transformer(torch.nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,


        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        d_classVector = 10


        self.decoder = Decoder(
            number_of_blocks=num_decoder_layers,
            d_model=dim_model,
            num_heads=num_heads,
            d_ff=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, classVector) -> torch.Tensor:
        return self.decoder(tgt, self.encoder(src), classVector)


























#TESTING ONLY
classVector = torch.rand(1, 10)
source = torch.rand(64 , 16, 512)
target = torch.rand(64, 16, 512)
#tfmr = Transformer()(source, target, classVector)

transformer = torch.nn.Transformer(d_model = 512, nhead = 8, num_encoder_layers = 6, num_decoder_layers = 6,
                                   dim_feedforward = 2048, dropout = 0.1,
                                   custom_decoder = Decoder)

#print(tfmr.shape)
#print(transformer.shape)








device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
generator = transformer.to(device)
num_epochs = 10
learning_rate = 0.01
batch_size = 10

#mnist = torchvision.datasets.MNIST("C:\\creative\\apps\\git\\CustomTransformer\\data\\", download = False)


#From the PyToch DCGAN example
dataset = torchvision.datasets.MNIST(root="C:\\creative\\apps\\git\\CustomTransformer\\data\\", download=True,
                           transform=torchvision.transforms.Compose([
                               #transforms.Resize(28),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5,), (0.5,)),
                           ]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)
img_list = []
optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
dimension_nz = 512


fixed_noise = torch.randn(batch_size, dimension_nz, 1, 1, device=device)

for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        classVector = torch.rand(1, 10)
        noise = torch.randn(batch_size, 1, dimension_nz, device=device)
        fake = generator(noise, noise, classVector)
        generator.zero_grad()


        if (i % 2 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))