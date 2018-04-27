import torch
from seq import Encoder, Decoder, run
from torch import Tensor, LongTensor
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from data import  Vocab, OpenSub, pad_batch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import parse, length_to_mask, masked_cross_entropy_loss, save_checkpoint
from tensorboardX import SummaryWriter
import os
import sys
import random
from time import strftime, localtime, time
args = parse()
torch.manual_seed(args.seed)
if args.cuda: torch.cuda.manual_seed(args.seed)






