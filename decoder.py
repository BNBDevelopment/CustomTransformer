import math

import torch
from attention import AddedNormalizedAttention
from attention import CrossAttention
from attention import MultiheadSelfRandGlobalAttention





#Positional Encoding
def postionalEncoding(sequence_len, d_model):
    pos_encoding_vector = torch.zeros(sequence_len, d_model)

    for position in range(sequence_len):
        for i in range(d_model):
            if i % 2 == 0:
                pos_encoding_vector[position, i] = math.sin(position / (10000 ** (2 * i / d_model)))
            else:
                pos_encoding_vector[position, i] = math.cos(position / (10000 ** (2 * i / d_model)))

    return pos_encoding_vector

def feed_forward(d_model, d_ff):
    #From the paper
    #d_model = 512
    #d_ff = 2048

    return torch.nn.Sequential(torch.nn.Linear(d_model, d_ff), torch.nn.ReLU(), torch.nn.Linear(d_ff, d_model))


#Output Embedding
class OutputEmbedder(torch.nn.Module):
    def __init__(self, d_in, embedding_size, d_model):
        #Default call
        super().__init__()

        #PyTorch has an embedder function built in
        #self.embed = torch.nn.Embedding(num_embeddings, d_model)

        self.embed_a = torch.nn.Linear(d_in, embedding_size)
        self.embed_b = torch.nn.Linear(1, d_model)

    def forward(self, outputs):
        #Per AIAYN "We also use the usual learned linear transformation [and softmax function] to convert the decoder output"
        first = self.embed_a(outputs)
        second = self.embed_b(first.unsqueeze(-1))
        return torch.softmax(second, dim=-1)


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
        prevBlockOutput = self.maskedMultiheadSelfRandGlobalAttention(cur_input, prevLayerOutput, prevLayerOutput)

        #Per AIAYN, "queries come from the previous decoder layer" and "memory keys and values come from the output of the encoder"
        #Should have Q = prev block output, K and V = "cross attention piece" = shaped_classVector?


        #TODO; restore this
        classVector = classVector.unsqueeze(-1)
        #cur_input = self.crossAttention(prevBlockOutput, classVector, classVector)
        cur_input = self.crossAttention(prevBlockOutput, classVector, classVector)
        return self.feedForward(cur_input)

        #return prevBlockOutput



class Decoder(torch.nn.Module):
    #Hyperparams as per AIAYN
    d_model = 512
    d_ff = 2048
    dropout = 0.1
    sequence_length = 784
    class_vector_length = 10

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

        #TODO: check this
        self.outputEmbedder = OutputEmbedder(self.class_vector_length, self.sequence_length, d_model)

        #Initialize the blocks
        block_list = []
        for count in range(self.number_of_blocks):
            block_list.append(Block(self.d_model, self.d_ff, self.num_heads, self.dropout, self.num_rand_glbl_tkns, self.cross_dimensions))
        self.layers = torch.nn.ModuleList(block_list)

        #Initialize final linear layer
        self.linear = torch.nn.Linear(self.d_model, self.d_model)

        #TODO #SOFTMAX



    def forward(self, classVector):

        #seq_len = input.size(1)
        #layer_count = input.size(2)

        seq_len = 784

        torch.nn.TransformerDecoder
        #Push forward through blocks, updating input after each step
        m_input = self.outputEmbedder(classVector)
        m_input = m_input + postionalEncoding(seq_len, self.d_model)
        output = m_input

        for blockLayer in self.layers:
            output = blockLayer(output, m_input, classVector)

        #return torch.softmax(self.linear(input), dim=-1)
        out_probs = torch.nn.functional.softmax(self.linear(output), dim=-1)
        return out_probs

