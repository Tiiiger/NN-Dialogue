import torch
from torch import LongTensor
from torch import Tensor
from torch.autograd import Variable

def length_to_mask(lengths, longest=None):
    if longest == None:
        longest = max(lengths)
    batch_size = len(lengths)
    index = torch.arange(0, longest).long()
    index = indewx.unsqueeze(0).expand(batch_size, longest)
    lengths = LongTensor(lengths).unsqueeze(0).expand(batch_size, longest)
    mask = index < lengths
    return mask

def masked_cross_entropy_loss(logits, max_length, lens):
    raise NotImplemented
