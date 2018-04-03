import torch
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from data import parse, OpenSub, pad_batch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def masked_cross_entropy_loss(logits, max_length, lens):
    raise NotImplemented

class Encoder(Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.embedding = torch.nn.Embedding(args.vocab_size, args.embed_size)
        self.lstm = torch.nn.LSTM(input_size=args.embed_size,
                                  hidden_size=args.hidden_size,
                                  num_layers=4,
                                  bidirectional=True)
    def forward(self, source, lens, hidden):
        dense = self.embedding(source)
        packed_dense = pack_padded_sequence(dense, lens)
        packed_outputs, hidden = self.lstm(packed_dense, hidden)
        def _cat(hidden):
            torch.cat(hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2], 2)
        hidden = tuple(_cat(h) for h in hidden)
        outputs = pad_packed_sequence(packed_outputs)
        return outupts, hidden

    def save_model(self, path):
        raise NotImplemented

class Attention(Module):

    def __init__(self):
        super(Attention, self).__init__()
        raise NotImplemented

    def forward(self, source_outputs):
        raise NotImplemented

    def backward(self, source_outputs):
        raise NotImplemented

class Decoder(Module):

    def __init__(self, params):
        super(Decoder, self).__init__()
        raise NotImplemented

    def forward(self):
        raise NotImplemented

args = parse()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

args.batch_size = 2

train_data = OpenSub(args)
test_data = OpenSub(args, "../Data/t_given_s_test.txt")
PAD = train_data.PAD
collate = lambda x:pad_batch(x, PAD)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, **kwargs)

encoder = Encoder(args).cuda() if args.cuda else Encoder(args)
decoder = Decoder(args).cuda() if args.cuda else Decoder(args)

encoder_optim = optim.SGD(encoder.parameters(), lr=args.lr)
decoder_optim = optim.SGD(decoder.parameters(), lr=args.lr)

def train(epoch):
    decoder.train()
    for batch_idx, (source, source_lens, target, target_lens) in enumerate(train_loader):
        source, target = Variable(source), Variable(target)
        if args.cuda():
            source, target = source.cuda(), target.cuda()
        encoder_outputs, encoder_last_hidden = encoder(source, source_lens, None)

def test():
    raise NotImplementedError

def evaluate():
    raise NotImplementedError

