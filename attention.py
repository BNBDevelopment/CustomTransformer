import random

import torch


class AttentionHead(torch.nn.Module):
    def __init__(self, dim_in, dim_q, dim_k, num_rand_glbl_tkns):
        super().__init__()
        self.query = torch.nn.Linear(dim_in, dim_q)
        self.key = torch.nn.Linear(dim_in, dim_k)
        self.value = torch.nn.Linear(dim_in, dim_k)
        self.num_rand_glbl_tkns = num_rand_glbl_tkns


    def forward(self, q, k, v):
        sdp = self.scaledDotProductAttention(self.query(q), self.key(k), self.value(v), self.num_rand_glbl_tkns)
        return sdp

    def createMask(self, lrlen, sequence_len, num_rand_glbl_tkns):
        #BigBird style random global attention tokens
        global_tokens= random.sample(range(0, lrlen), num_rand_glbl_tkns)

        mask2 = torch.zeros(sequence_len, lrlen)
        for i in range(0,lrlen):
            for j in range(0, sequence_len):
                #Exclude
                if i < j and i not in global_tokens and j not in global_tokens:
                #if i < j and i not in global_tokens_r and j not in global_tokens_c:
                    mask2[i, j] = torch.tensor(float('-inf'))
        return mask2


    def scaledDotProductAttention(self, Q, K, V, num_rand_glbl_tkns):
        #num = torch.dot(Q, K.transpose(1,2))
        num = torch.matmul(Q, K.transpose(1,2))
        denom = K.size(-1) ** 0.5

        #Should be -1 instead of two to consistently get last?
        mask = self.createMask(num.size(-1),num.size(-2), num_rand_glbl_tkns)
        num = num + mask

        #softmax = torch.nn.softmax(num / denom)
        softmax = torch.nn.functional.softmax(num/denom, dim=-1)
        #attentionQKV = torch.dot(softmax, V)
        attentionQKV = torch.matmul(softmax, V)
        return attentionQKV



class MultiheadAttention(torch.nn.Module):


    def __init__(self, number_of_heads, model_dimensions, seq_dimensions, num_rand_glbl_tkns):
        super().__init__()


        attentionHeads = []
        for i in range(number_of_heads):
            attentionHeads.append(AttentionHead(model_dimensions, seq_dimensions, seq_dimensions, num_rand_glbl_tkns))
        self.heads = torch.nn.ModuleList(attentionHeads)

        self.linear = torch.nn.Linear(number_of_heads * seq_dimensions, model_dimensions)

    def forward(self, Q, K, V):
        head_values = []
        for head in self.heads:
            head_values.append(head.forward(Q, K, V))

        #USing concatention NOT summation
        output = self.linear(torch.concat(head_values,dim=-1))
        return output


class AddedNormalizedAttention(torch.nn.Module):
    def __init__(self, layer, dimensions, dropout_value):
        super().__init__()
        self.layer = layer
        self.normalizer = torch.nn.LayerNorm(dimensions)
        self.dropout = torch.nn.Dropout(dropout_value)

    def forward(self, *tensors):
        return self.normalizer(tensors[0] + self.dropout(self.layer(*tensors)))