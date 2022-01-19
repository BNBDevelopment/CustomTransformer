import random

import torch


class AttentionHead(torch.nn.Module):
    def __init__(self, dim_in, dim_q, dim_k):
        super().__init__()
        self.query = torch.nn.Linear(dim_in, dim_q)
        self.key = torch.nn.Linear(dim_in, dim_k)
        self.value = torch.nn.Linear(dim_in, dim_k)


    def forward(self, q, k, v):
        sdp = self.scaledDotProductAttention(self.query(q), self.key(k), self.value(v))
        return sdp

    def createMask(self, lrlen, sequence_len):
        #mask = (torch.triu(torch.ones(lrlen, sequence_len)) == 1).transpose(0, 1)
        #mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        #return mask
        #print(mask)

        #BigBird style random global attention tokens
        global_tokens_r = random.sample(range(0, lrlen), 3)
        global_tokens_c = random.sample(range(0, sequence_len), 3)
        gt_index_pairs = []
        for index in range(3):
            gt_index_pairs.append((global_tokens_r[index],global_tokens_c[index]))


        mask2 = torch.zeros(sequence_len, lrlen)
        for i in range(0,lrlen):
            for j in range(0, sequence_len):
                #Exclude
                if i < j and (i,j) not in gt_index_pairs:
                #if i < j and i not in global_tokens_r and j not in global_tokens_c:
                    mask2[i, j] = torch.tensor(float('-inf'))
        return mask2


    def scaledDotProductAttention(self, Q, K, V):
        #num = torch.dot(Q, K.transpose(1,2))
        num = torch.matmul(Q, K.transpose(1,2))
        denom = K.size(-1) ** 0.5

        #Should be -1 instead of two to consistently get last?
        mask = self.createMask(num.size(-1),num.size(-2))
        num = num + mask

        #softmax = torch.nn.softmax(num / denom)
        softmax = torch.nn.functional.softmax(num/denom, dim=-1)
        #attentionQKV = torch.dot(softmax, V)
        attentionQKV = torch.matmul(softmax, V)
        return attentionQKV



class MultiheadAttention(torch.nn.Module):


    def __init__(self, number_of_heads, model_dimensions, seq_dimensions):
        super().__init__()


        attentionHeads = []
        for i in range(number_of_heads):
            attentionHeads.append(AttentionHead(model_dimensions, seq_dimensions, seq_dimensions))
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