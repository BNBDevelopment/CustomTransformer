import random
import torch

global_tokens = None
mask = None

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

    #TODO: Fix masking
    def createMask(self, lrlen, sequence_len, num_rand_glbl_tkns):
        #BigBird style random global attention tokens
        global global_tokens

        if global_tokens == None:
            global_tokens = random.sample(range(0, lrlen), num_rand_glbl_tkns)

        mask = torch.triu(torch.ones(sequence_len, sequence_len)).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        for selected_global_token in global_tokens:
            mask[:,selected_global_token] = float(0.0)
            mask[selected_global_token,:] = float(0.0)
        return mask


    def scaledDotProductAttention(self, Q, K, V, num_rand_glbl_tkns):
        #Scaled Dot Product formula as per AIAYN
        num = torch.matmul(Q, K.transpose(1,-1))
        denom = K.size(-1) ** 0.5

        #Create and apply masking
        global mask
        if mask == None:
            mask = self.createMask(num.size(-1),num.size(-2), num_rand_glbl_tkns)
        num = num + mask

        softmax = torch.nn.functional.softmax(num/denom, dim=-1)
        attentionQKV = torch.matmul(softmax, V)
        return attentionQKV



class MultiheadSelfRandGlobalAttention(torch.nn.Module):
    def __init__(self, number_of_heads, model_dimensions, seq_dimensions, num_rand_glbl_tkns):
        #Default call
        super().__init__()

        # Create desired number of attention heads
        attentionHeads = []
        for i in range(number_of_heads):
            attentionHeads.append(AttentionHead(model_dimensions, seq_dimensions, seq_dimensions, num_rand_glbl_tkns))
        self.heads = torch.nn.ModuleList(attentionHeads)

        self.linear = torch.nn.Linear(number_of_heads * seq_dimensions, model_dimensions)


    def forward(self, Q, K, V):
        #Push forward through attention head mechanisms
        head_values = []
        for head in self.heads:
            head_values.append(head.forward(Q, K, V))

        #Concatenating attention head layers together, then putting them through Linear layer as per AIAYN
        shapedConcat = torch.concat(head_values,dim=-1)
        output = self.linear(shapedConcat)

        return output


class AddedNormalizedAttention(torch.nn.Module):
    def __init__(self, layer, dimensions, dropout_value):
        #Default call
        super().__init__()

        self.layer = layer
        self.normalizer = torch.nn.LayerNorm(dimensions)
        self.dropout = torch.nn.Dropout(dropout_value)


    def forward(self, *tensors):
        after_dropout = self.dropout(self.layer(*tensors))
        cur_matrix = tensors[0]

        #keeping shape consistent between two attention types
        if cur_matrix.size(0) == after_dropout.size(1) and cur_matrix.size(1) == after_dropout.size(0):
            after_dropout = after_dropout.transpose(0,1)

        return self.normalizer(cur_matrix + after_dropout)



class CrossAttentionHead(torch.nn.Module):
    def __init__(self, dim_in, dim_q):
        # Default call
        super().__init__()

        self.query = torch.nn.Linear(dim_in, dim_q)
        self.key = torch.nn.Linear(dim_in, dim_q)
        self.value = torch.nn.Linear(dim_in, dim_q)


    def forward(self, q, k, v):
        sdp = self.scaledDotProductAttention(self.query(q), self.key(k), self.value(v))
        return sdp


    def scaledDotProductAttention(self, Q, K, V):
        # Scaled Dot Product formula as per AIAYN
        num = torch.matmul(Q, K.transpose(0,-1))
        denom = K.size(-1) ** 0.5
        softmax = torch.nn.functional.softmax(num/denom, dim=-1)
        attentionQKV = torch.matmul(softmax, V)
        return attentionQKV



class CrossAttention(torch.nn.Module):
    def __init__(self, number_of_heads, model_dimensions, seq_dimensions):
        # Default call
        super().__init__()

        #Create desired number of attention heads
        attentionHeads = []
        for i in range(number_of_heads):
            attentionHeads.append(CrossAttentionHead(model_dimensions, seq_dimensions))
        self.heads = torch.nn.ModuleList(attentionHeads)

        self.linear = torch.nn.Linear(number_of_heads * seq_dimensions, model_dimensions)


    def forward(self, Q, K, V):
        # Push forward through attention head mechanisms
        head_values = []
        for head in self.heads:
            head_values.append(head.forward(Q, K, V))

        #Concatenating attention head layers together, then putting them through Linear layer as per AIAYN
        shapedConcat = torch.concat(head_values,dim=-1)
        output = self.linear(shapedConcat)
        #output = self.linear(torch.concat(head_values,dim=-1))
        return output