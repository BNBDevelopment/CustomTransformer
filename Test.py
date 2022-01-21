import torch


#Scaled Dot Product Function
def sdp(Q, K, V):
    temp = Q.bmm(K.transpose(1,2))
    scale = Q.size(-1) ** 0.05
    softmax = torch.nn.functional.softmax(temp / scale, dim=-1)
    return softmax.bmm(V)

def sdpAttention(Q,K,V):

    num = torch.dot(Q,K.transpose())
    denom = K.size(-1) ** 0.5
    softmax = torch.nn.softmax(num / denom)
    attentionQKV = torch.dot(softmax,V)
    return attentionQKV


#TODO
def postionalEncoding(sequence_len, layer_count):
    pos_encoding_vector = torch.zeros(sequence_len)

    for position in range(sequence_len):
        for i in range(layer_count):
            if i % 2 == 0:
                pos_encoding_vector[position, i] = torch.sin(position/(10000**(2*i /layer_count)))
            else:
                pos_encoding_vector[position, i] = torch.cos(position/(10000**(2*i /layer_count)))



class AttentionHead(torch.nn.Module):
    def __init__(self, dim_in, dim_q, dim_k):
        super().__init__()
        self.query = torch.nn.Linear(dim_in, dim_q)
        self.key = torch.nn.Linear(dim_in, dim_k)
        self.value = torch.nn.Linear(dim_in, dim_k)

    def forward(self, q, k, v):
        sdp = sdpAttention(self.query(q), self.key(k), self.value(v))






class MultiheadAttention(torch.nn.Module):

    #SET NUMBER OF ATTENTION HEADS HERE
    number_attention_heads = 6
                           #dim_in          #dim_q         #dim_k
    def __init__(self, hidden_dimensions, num_layers, output_dimensions):
        super().__init__()


        attentionHeads = [AttentionHead(hidden_dimensions, num_layers, output_dimensions)]
        for i in range(self.number_attention_heads):
            attentionHeads.append()
        self.heads = torch.nn.ModuleList(attentionHeads)

        self.linear = torch.nn.Linear(self.number_attention_heads * output_dimensions * hidden_dimensions)

    def forward(self, Q, K, V):
        head_values = []
        for head in self.heads:
            head_values.append(head(Q, K, V))

        #USing concatention NOT summation
        torch.concat(head_values)



def feed_forward():

    #From the paper
    d_model = 512
    d_ff = 2048

    return torch.nn.Sequential(torch.nn.Linear(d_model, d_ff), torch.nn.ReLU(), torch.nn.Linear(d_ff, d_model))



class Residual(torch.nn.Module)
    def __init__(self):

    def forward(self):



class Block(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.maskedMultiHeadAttention = Residual(MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),dimension=dim_model,dropout=dropout,)

        self.multiHeadAttention = Residual(MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),dimension=dim_model,dropout=dropout,)

        self.feedForward = Residual(feed_forward(dim_model, dim_feedforward),dimension=dim_model,dropout=dropout,)

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
            block_list.append(Block(self.dim_model, self.num_heads, self.dim_feedforward, self.dropout))

        self.layers = torch.nn.ModuleList(block_list)

        self.linear = torch.nn.Linear(self.dim_model, self.dim_model)

    def forward(self, target, memory):
        seq_len, dimension = target.size(1), target.size(2)
        target += position_encoding(seq_len, dimension)
        for layer in self.layers:
            tgt = layer(tgt, memory)

        return torch.softmax(self.linear(tgt), dim=-1)