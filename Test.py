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




class AttentionHead(torch.nn.Module):
    def __init__(self, dim_in, dim_q, dim_k):
        super().__init__()
        self.query = torch.nn.Linear(dim_in, dim_q)
        self.key = torch.nn.Linear(dim_in, dim_k)
        self.value = torch.nn.Linear(dim_in, dim_k)

    def forward(self, q, k, v):
        sdp = sdpAttention(self.query(q), self.key(k), self.value(v))



class MultiheadAttention(torch.nn.Module):
                           #dim_in          #dim_q         #dim_k
    def __init__(self, hidden_dimensions, num_layers, output_dimensions):
        super().__init__()

        number_attention_heads = 6
        attentionHeads = [AttentionHead(hidden_dimensions, num_layers, output_dimensions)]
        for i in range(number_attention_heads):
            attentionHeads.append()
        self.heads = torch.nn.ModuleList(attentionHeads)

        self.linear = torch.nn.Linear(number_attention_heads * output_dimensions * hidden_dimensions)

    def forward(self, Q, K, V):
        head_values = []
        for head in self.heads:
            head_values.append(head(Q, K, V))

        #USing concatention NOT summation
        torch.concat(head_values)



class Decoder(torch.nn.Module):
    def __init__(self):

    def forward(self):


class DecoderBlock(torch.nn.Module)