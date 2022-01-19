import torch


class AttentionHead(torch.nn.Module):
    def __init__(self, dim_in, dim_q, dim_k):
        super().__init__()
        self.query = torch.nn.Linear(dim_in, dim_q)
        self.key = torch.nn.Linear(dim_in, dim_k)
        self.value = torch.nn.Linear(dim_in, dim_k)


    def forward(self, q, k, v):
        sdp = self.scaledDotProductAttention(self.query(q), self.key(k), self.value(v))


    def scaledDotProductAttention(Q, K, V):
        num = torch.dot(Q, K.transpose())
        denom = K.size(-1) ** 0.5
        softmax = torch.nn.softmax(num / denom)
        attentionQKV = torch.dot(softmax, V)
        return attentionQKV



class MultiheadAttention(torch.nn.Module):


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
        self.linear(torch.concat(head_values))


class AddedNormalizedAttention(torch.nn.Module):
    def __init__(self, dimensions, dropout_value):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dimensions)
        self.dropput = torch.nn.Dropout(dropout_value)

    def forward(self, *tensors):
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))