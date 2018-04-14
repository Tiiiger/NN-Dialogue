import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from data import  OpenSub, pad_batch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import parse, length_to_mask, masked_cross_entropy_loss

class Encoder(Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size+4, args.embed_size)
        # Only accept 4 layers bi-directional LSTM right now
        self.rnn = nn.LSTM(input_size=args.embed_size,
                                  hidden_size=args.hidden_size,
                                  num_layers=4,
                                  bidirectional=True)
    def forward(self, source, lens, hidden=None):
        dense = self.embedding(source)
        packed_dense = pack_padded_sequence(dense, lens)
        packed_outputs, hidden = self.rnn(packed_dense, hidden)
        def _cat(hidden):
            return torch.cat((hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]), 2)
        hidden = tuple(_cat(h) for h in hidden)
        outputs, output_lens = pad_packed_sequence(packed_outputs)
        return outputs, hidden

    def save_model(self, path):
        torch.save(self, PATH)

class Attention(Module):

    def __init__(self, args):
        super(Attention, self).__init__()
        self.score = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

    def forward(self, decoder_outputs, encoder_outputs, source_lengths):
        """
        Return attention scores.
        args:
        decoder_outputs: TxBx*
        encoder_outputs: TxBx*
        returns:
        attention scores: Bx1xT
        """
        projected_encoder_outputs = self.score(encoder_outputs) \
                                        .permute(1, 2, 0) # batch first
        decoder_outputs = decoder_outputs.transpose(0,1)
        scores = decoder_outputs.bmm(projected_encoder_outputs)
        scores = scores.squeeze(1)
        mask = length_to_mask(source_lengths, source_lengths[0])
        scores.data.masked_fill_(mask, float('-inf'))
        scores = F.softmax(scores, dim=1)
        return scores.unsqueeze(1)

class Decoder(Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(args.vocab_size+4, args.embed_size)
        self.rnn = nn.LSTM(input_size = args.embed_size,
                           hidden_size = 2 * args.hidden_size,
                           num_layers = 4,
                           bidirectional=False)
        self.output = nn.Linear(4*args.hidden_size, args.hidden_size)
        self.predict = nn.Linear(args.hidden_size, args.vocab_size)
        self.attention = Attention(args)

    def forward(self, target, encoder_outputs, source_lengths, hidden=None):
        """
        args:
        target: A LongTensor contains a word of the target sentence. size: 1*B.
        """
        embed_target = self.embed(target)
        decoder_outputs, decoder_hiddens = self.rnn(embed_target, hidden)
        atten_scores = self.attention(decoder_outputs,
                                      encoder_outputs, source_lengths)
        context = atten_scores.bmm(encoder_outputs.transpose(0,1))
        concat = torch.cat([context, decoder_outputs.transpose(0,1)], -1)
        atten_outputs = F.tanh(self.output(concat))
        predictions = self.predict(atten_outputs)
        predictions.squeeze_(1)
        return predictions, decoder_hiddens, atten_scores


args = parse()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_data = OpenSub(args)
test_data = OpenSub(args, "../Data/t_given_s_test.txt")
PAD = train_data.PAD
collate = lambda x:pad_batch(x, PAD)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True, collate_fn=collate, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=args.batch_size,
                                          shuffle=True, collate_fn=collate, **kwargs)

encoder = Encoder(args).cuda() if args.cuda else Encoder(args)
decoder = Decoder(args).cuda() if args.cuda else Decoder(args)

encoder_optim = optim.SGD(encoder.parameters(), lr=args.lr)
decoder_optim = optim.SGD(decoder.parameters(), lr=args.lr)

def train(epoch):
    encoder.train()
    decoder.train()
    for batch_idx, (source, source_lens, target, target_lens) in enumerate(train_loader):
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        source, target = Variable(source), Variable(target)
        if args.cuda:
            source, target = source.cuda(), target.cuda()
        encoder_outputs, encoder_last_hidden = encoder(source, source_lens, None)
        max_target_len = max(target_lens)
        decoder_outputs = torch.zeros(max_target_len, args.batch_size, args.vocab_size) # preallocate
        decoder_hidden = encoder_last_hidden
        target_slice = Variable(torch.zeros(1, args.batch_size).fill_(train_data.SOS).long())
        for l in range(max_target_len):
            predictions, decoder_hiddens, atten_scores = decoder(target_slice, encoder_outputs, source_lens, decoder_hidden)
            raise NotImplementedError
        raise NotImplementedError

def test():
    raise NotImplementedError

def evaluate():
    raise NotImplementedError


train(1)
