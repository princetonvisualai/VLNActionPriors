import torch
from torch import nn
from collections import OrderedDict
import math
import torch.nn.functional as F
import numpy as np


class TranslatorTCN(nn.Module):
    """
    A module that runs multiple steps of multi-head attention layers
    """
    def __init__(self, hidden_size, vocab_size=None, share_embedding=False, use_end_token=False):
        super(TranslatorTCN, self).__init__()
        self.vocab_size = vocab_size
        self.kernel_size = 3
        self.stride = 2
        self.pad = 1
        self.tcn = nn.Conv1d(hidden_size, hidden_size, self.kernel_size, self.stride, self.pad)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, seq_lengths):
        # concat inputs and masks
        input = input.transpose(1,2)
        h = self.tcn(input)
        h = h.transpose(1,2)
        logit = self.output_layer(h)
        pseodu_seq_lengths = []

        for i in range(len(seq_lengths)):
          pseodu_seq_lengths.append(seq_lengths[i] // self.stride)

        return logit, pseodu_seq_lengths

class TranslatorTCNSkip(nn.Module):
    """
    A module that runs multiple steps of multi-head attention layers
    """
    def __init__(self, hidden_size, vocab_size=None, share_embedding=False, use_end_token=False):
        super(TranslatorTCNSkip, self).__init__()
        self.vocab_size = vocab_size

        self.tcn = nn.Conv1d(hidden_size, hidden_size, 1, 1)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, seq_lengths, skip=True):
        # concat inputs and masks
        if skip:
          input = input.transpose(1,2)
          h = self.tcn(input)
          h = h.transpose(1,2)
          logit = self.output_layer(h)
        else:
          logit = input
        pseodu_seq_lengths = []

        for i in range(len(seq_lengths)):
          pseodu_seq_lengths.append(seq_lengths[i])

        return logit, pseodu_seq_lengths
