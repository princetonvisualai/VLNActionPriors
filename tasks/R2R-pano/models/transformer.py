import torch
from torch import nn
from collections import OrderedDict
import math
import torch.nn.functional as F
import numpy as np


class DecoderTransformer(nn.Module):
    """
    A module that runs multiple steps of multi-head attention layers
    """
    def __init__(self, hidden_size, num_heads=8, num_layers=1,
                 dropout_ratio=0, vocab_size=None, share_embedding=False):
        super(DecoderTransformer, self).__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.tfs = []
        for layer in range(num_layers):
            self.tfs += [MultiheadAttentionLayer(hidden_size, num_heads, hidden_size)]
        self.tfs = nn.ModuleList(self.tfs)

        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, mask, return_full=False):
        """
          input tensors:
            input      -> [N, D]
            input_mask -> [N, 1]
              
          output tensors:
            logit      -> [N, D]
        """

        # concat inputs and masks
        h     = input
        for n in range(0, self.num_layers):
            h = self.tfs[n](h, h, h, mask)
        if not return_full:
          logit = self.output_layer(h[:,-1,:])
        else:
          logit = self.output_layer(h)

        return logit


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(1).expand(scores.shape)
        scores = scores.masked_fill(mask == 0, -np.inf)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output, scores


class MultiheadAttentionLayer(nn.Module):
    def __init__(self, d_model, heads, d_hid, dropout=0.1):
        super(MultiheadAttentionLayer, self).__init__()

        self.d_model = d_model
        self.d_hid   = d_hid
        self.d_k = d_hid // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_hid)
        self.v_linear = nn.Linear(d_model, d_hid)
        self.k_linear = nn.Linear(d_model, d_hid)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Linear(d_hid, d_hid)
        self.norm = nn.LayerNorm(d_hid)
        self.out = nn.Linear(d_hid, d_hid)

    def forward(self, q, k, v, mask=None):
        bs, length = q.size(0), k.size(1)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        weighted_v, scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = weighted_v.transpose(1,2).contiguous().view(-1, self.d_hid)

        #concat_norm = self.norm(self.ff(concat.squeeze())).view(bs, -1, self.d_hid)
        concat_norm = self.ff(concat.squeeze()).view(bs, -1, self.d_hid)
        #output = self.out(concat_norm)
        output = concat_norm
        if mask is not None:
          output = output * mask.float().unsqueeze(-1).expand(output.shape)

        return output


if __name__ == "__main__":

    batch_size = 64
    input_size = 128
    hidden_size = 256
    T = 13
    decoder_kwargs = {"input_size": input_size,
                      "hidden_size": hidden_size,
                      "num_heads": 8,
                      "num_layers": 1,
                      "dropout": 0,
                     }
    model = DecoderTransformer(**decoder_kwargs).cuda()

    inputs = torch.randn(batch_size, T, input_size).cuda()
    mask   = torch.ones(batch_size, T).cuda()

    output = model(inputs, mask)

    import pdb; pdb.set_trace()
