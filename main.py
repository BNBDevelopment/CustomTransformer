import torch
from attention import AddedNormalizedAttention
from attention import MultiheadAttention

#Output Embedding


#Positional Encoding


def postionalEncoding(sequence_len, layer_count):
    pos_encoding_vector = torch.zeros(sequence_len)

    for position in range(sequence_len):
        for i in range(layer_count):
            if i % 2 == 0:
                pos_encoding_vector[position, i] = torch.sin(position / (10000 ** (2 * i / layer_count)))
            else:
                pos_encoding_vector[position, i] = torch.cos(position / (10000 ** (2 * i / layer_count)))


def feed_forward(d_model, d_ff):
    #From the paper
    #d_model = 512
    #d_ff = 2048

    return torch.nn.Sequential(torch.nn.Linear(d_model, d_ff), torch.nn.ReLU(), torch.nn.Linear(d_ff, d_model))



class Block(torch.nn.Module):

    def __init__(self, d_model, d_ff, num_heads, dropout):
        super().__init__()

        dimensions_per_head = max(d_model // num_heads, 1)

        self.maskedMultiHeadAttention = AddedNormalizedAttention(MultiheadAttention(num_heads, d_model, dimensions_per_head, dimensions_per_head), d_model, dropout)

        self.multiHeadAttention = AddedNormalizedAttention(MultiheadAttention(num_heads, d_model, dimensions_per_head, dimensions_per_head), d_model, dropout)

        self.feedForward = AddedNormalizedAttention(feed_forward(d_model, d_ff), d_model, dropout)

    def forward(self, target, memory):
        tgt = self.attention_1(target, target, target)
        tgt = self.attention_2(tgt, memory, memory)
        return self.feed_forward(tgt)



class Decoder(torch.nn.Module):
    d_model = 512
    d_ff = 2048

    num_heads = 6
    number_of_blocks = 6

    dim_feedforward = 2048
    dropout = 0.1

    def __init__(self):
        super().__init__()

        block_list = []
        for count in self.number_of_blocks:
            block_list.append(Block(self.d_model, self.d_ff, self.num_heads, self.dropout))

        self.layers = torch.nn.ModuleList(block_list)

        self.linear = torch.nn.Linear(self.dim_model, self.dim_model)

    def forward(self, target, memory):
        seq_len, layer_count = target.size(1), target.size(2)
        target += postionalEncoding(seq_len, layer_count)
        for layer in self.layers:
            tgt = layer(tgt, memory)

        return torch.softmax(self.linear(tgt), dim=-1)

