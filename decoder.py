import math

import torch
from attention import AddedNormalizedAttention
from attention import CrossAttention
from attention import MultiheadSelfRandGlobalAttention





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


#Output Embedding
class OutputEmbedder(torch.nn.Module):
    def __init__(self, num_embeddings, d_model):
        #Default call
        super().__init__()

        #PyTorch has an embedder function built in
        #self.embed = torch.nn.Embedding(num_embeddings, d_model)

        self.embed = torch.nn.Linear(d_model, num_embeddings)

    def forward(self, outputs):
        #Per AIAYN "We also use the usual learned linear transformation [and softmax function] to convert the decoder output"
        return torch.softmax(self.embed(outputs), dim=-1)


class Block(torch.nn.Module):

    def __init__(self, d_model, d_ff, num_heads, dropout, num_rand_glbl_tkns, cross_dimensions):
        super().__init__()

        seq_dimensions = max(d_model // num_heads, 1)
        self.d_model = d_model

        #Block Components Below
        self.maskedMultiheadSelfRandGlobalAttention = AddedNormalizedAttention(MultiheadSelfRandGlobalAttention(num_heads, d_model, seq_dimensions, num_rand_glbl_tkns), d_model, dropout)

        self.crossAttention = AddedNormalizedAttention(CrossAttention(num_heads, d_model, seq_dimensions, cross_dimensions), d_model, dropout)

        self.feedForward = AddedNormalizedAttention(feed_forward(d_model, d_ff), d_model, dropout)

    def forward(self, cur_input, prevLayerOutput, classVector):

        #reshaping the class vector for later matrix manipulations
        shaped_classVector = classVector.transpose(0,1)
        #shaped_classVector = shaped_classVector.expand(shaped_classVector.shape[0], self.d_model)
        shaped_classVector = shaped_classVector.expand(self.d_model, shaped_classVector.shape[1])

        #shaped_classVector2 = classVector.transpose(0, 1)
        shaped_classVector2 = classVector.expand(classVector.shape[0], self.d_model)


        #prevBlockOutput = self.maskedMultiheadSelfRandGlobalAttention(cur_input, cur_input, cur_input)
        prevBlockOutput = self.maskedMultiheadSelfRandGlobalAttention(cur_input, prevLayerOutput, prevLayerOutput)

        #Per AIAYN, "queries come from the previous decoder layer" and "memory keys and values come from the output of the encoder"
        #Should have Q = prev block output, K and V = "cross attention piece" = shaped_classVector?
        #cur_input = self.crossAttention(shaped_classVector, prevBlockOutput, prevBlockOutput)

        #cur_input = self.crossAttention(prevBlockOutput, shaped_classVector, shaped_classVector)
        cur_input = self.crossAttention(prevBlockOutput, shaped_classVector2, shaped_classVector2)
        return self.feedForward(cur_input)



class Decoder(torch.nn.Module):
    #Hyperparams as per AIAYN
    d_model = 512
    d_ff = 2048
    dropout = 0.1

    #Custom Hyper Params
    num_heads = 6
    number_of_blocks = 6
    num_rand_glbl_tkns = 3

    def __init__(self, d_model, d_ff, num_heads, number_of_blocks, dropout, cross_dimensions):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_heads = num_heads
        self.number_of_blocks = number_of_blocks
        self.cross_dimensions = cross_dimensions

        self.outputEmbedder = OutputEmbedder(d_model, d_model)

        #Initialize the blocks
        block_list = []
        for count in range(self.number_of_blocks):
            block_list.append(Block(self.d_model, self.d_ff, self.num_heads, self.dropout, self.num_rand_glbl_tkns, self.cross_dimensions))
        self.layers = torch.nn.ModuleList(block_list)

        #Initialize final linear layer
        self.linear = torch.nn.Linear(self.d_model, 784)


    #tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
    #tgt_key_padding_mask=tgt_key_padding_mask,
    #memory_key_padding_mask=memory_key_padding_mask

    #def forward(self, input, layerOutput, classVector):
    def forward(self, tgt, memory, tgt_mask = None,
                    memory_mask = None, tgt_key_padding_mask = None,
                    memory_key_padding_mask = None):

        #hijacking parameters to allow for integration into PyTorch infrustructure
        input = tgt
        layerOutput = memory
        classVector = tgt_mask

        #Class vector is only used for cross attention mechanism
        #MNIST class vector dimensions = 10?
        seq_len = input.size(1)
        layer_count = input.size(2)

        #Push forward through blocks, updating input after each step
        input = self.outputEmbedder(input)
        input = input + postionalEncoding(seq_len, layer_count)
        for blockLayer in self.layers:
            input = blockLayer(input, layerOutput, classVector)

        return torch.softmax(self.linear(input), dim=-1)

