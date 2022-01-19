import math

import torch
from attention import AddedNormalizedAttention
from attention import MultiheadAttention

#Output Embedding


#Positional Encoding


def postionalEncoding(sequence_len, layer_count):
    pos_encoding_vector = torch.zeros(sequence_len, layer_count)

    for position in range(sequence_len):
        for i in range(layer_count):
            if i % 2 == 0:
                pos_encoding_vector[position, i] = math.sin(position / (10000 ** (2 * i / layer_count)))
            else:
                pos_encoding_vector[position, i] = math.cos(position / (10000 ** (2 * i / layer_count)))

    return pos_encoding_vector

def feed_forward(d_model, d_ff):
    #From the paper
    #d_model = 512
    #d_ff = 2048

    return torch.nn.Sequential(torch.nn.Linear(d_model, d_ff), torch.nn.ReLU(), torch.nn.Linear(d_ff, d_model))



class Block(torch.nn.Module):

    def __init__(self, d_model, d_ff, num_heads, dropout):
        super().__init__()

        seq_dimensions = max(d_model // num_heads, 1)

        self.maskedMultiHeadAttention = AddedNormalizedAttention(MultiheadAttention(num_heads, d_model, seq_dimensions), d_model, dropout)

        self.multiHeadAttention = AddedNormalizedAttention(MultiheadAttention(num_heads, d_model, seq_dimensions), d_model, dropout)

        self.feedForward = AddedNormalizedAttention(feed_forward(d_model, d_ff), d_model, dropout)

    def forward(self, cur_input, memory):
        cur_input = self.maskedMultiHeadAttention(target, target, target)
        cur_input = self.multiHeadAttention(cur_input, memory, memory)
        return self.feedForward(cur_input)



class Decoder(torch.nn.Module):
    d_model = 512
    d_ff = 2048

    num_heads = 6
    number_of_blocks = 6

    dropout = 0.1

    def __init__(self, d_model, d_ff, num_heads, number_of_blocks, dropout):
        super().__init__()

        block_list = []
        for count in range(self.number_of_blocks):
            block_list.append(Block(self.d_model, self.d_ff, self.num_heads, self.dropout))

        self.layers = torch.nn.ModuleList(block_list)

        self.linear = torch.nn.Linear(self.d_model, self.d_model)

    def forward(self, target, layerOutput):
        seq_len, layer_count = target.size(1), target.size(2)
        target += postionalEncoding(seq_len, layer_count)
        for blockLayer in self.layers:
            target = blockLayer(target, layerOutput)

        return torch.softmax(self.linear(target), dim=-1)























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
        src += position_encoding(seq_len, dimension)
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
        self.decoder = Decoder(
            number_of_blocks=num_decoder_layers,
            d_model=dim_model,
            num_heads=num_heads,
            d_ff=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return self.decoder(tgt, self.encoder(src))

#TESTING ONLY
source = torch.rand(64 , 16, 512)
target = torch.rand(64, 16, 512)
tfmr = Transformer()(source, target)
print(tfmr.shape)