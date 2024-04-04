import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math



class PrepareForMultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()

        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        batchSize, sequSize, featureSize = x.shape
        x = self.linear(x)
        x = x.view(batchSize, self.heads, sequSize, self.d_k)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()

        self.d_k = d_model#  // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(self.d_k*self.heads, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None
    

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        mask = mask.unsqueeze(-1)
        return mask
    
    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention




    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        values, attn = self.scaled_dot_product(query, key, value)
        x = values.reshape(seq_len, batch_size, -1)
        x = self.output(x)

        return x


class Attention_Block_defAttentionSequSize(nn.Module):

    def __init__(self, blocksize: int, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()

        assert blocksize%4 == 0, "Blocksize must be divisible by 4"

        self.blocksize = blocksize
        self.heads = heads
        self.d_model = d_model
        self.dropout_prob = dropout_prob
        self.bias = bias

        self.MHA = MultiHeadAttention(heads=self.heads, d_model=self.d_model, dropout_prob=self.dropout_prob, bias=self.bias)
        self.layernorm1 = nn.LayerNorm(self.d_model)

        self.feed_forward = nn.Linear(self.d_model, self.d_model)
        self.layernorm2 = nn.LayerNorm(self.d_model)
    
    def forward(self, query, key, value):

        # Reshape data
        query_res = list()
        key_res   = list()
        value_res = list()
        batchNum  = len(query)
        blocknum  = int((query.shape[1]/self.blocksize)*2 - 1)
        if blocknum>1:
            for bIdx in range(batchNum):
                for blockIdx in range(blocknum):
                    startIdx = blockIdx*int(self.blocksize/2)
                    endIdx   = blockIdx*int(self.blocksize/2) + self.blocksize
                    query_res.append(query[bIdx,startIdx:endIdx])
                    key_res.append(key[bIdx,startIdx:endIdx])
                    value_res.append(value[bIdx,startIdx:endIdx])
            query_res = torch.stack(query_res)
            key_res   = torch.stack(key_res)
            value_res = torch.stack(value_res)
        else:
            query_res = query
            key_res   = key
            value_res = value

        # Calc attention results
        x_MHA = self.MHA(query=query_res, key=key_res, value=value_res)
        x = x_MHA+value_res
        x = self.layernorm1(x)

        x_FF = self.feed_forward(x)
        x = x_FF+x
        x = self.layernorm2(x)

        # Unreshape
        if blocknum>1:
            recover_x = list()
            startIdx = int(0.25*self.blocksize)
            endIdx   = int(self.blocksize-startIdx)
            for bIdx in range(len(query)):
                recover_x_ = [x[bIdx*blocknum,:endIdx]]
                for blockIdx in range(1,blocknum-1):
                    recover_x_.append(x[bIdx*blocknum+blockIdx,startIdx:endIdx])
                recover_x_.append(x[(bIdx+1)*blocknum-1,startIdx:])
                recover_x.append(torch.cat(recover_x_,0))
            recover_x = torch.stack(recover_x)
        else:
            recover_x = x

        return recover_x
