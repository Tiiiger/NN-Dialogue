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
        self.embedding = nn.Embedding(args.vocab_size, args.embed_size)
        self.lstm = nn.LSTM(input_size=args.embed_size,
                                  hidden_size=args.hidden_size,
                                  num_layers=4,
                                  bidirectional=True)
    def forward(self, source, lens, hidden=None):
        dense = self.embedding(source)
        packed_dense = pack_padded_sequence(dense, lens)
        packed_outputs, hidden = self.lstm(packed_dense, hidden)
        def _cat(hidden):
            torch.cat(hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2], 2)
        hidden = tuple(_cat(h) for h in hidden)
        outputs = pad_packed_sequence(packed_outputs)
        return outupts, hidden

    def save_model(self, path):
        torch.save(self, PATH)

class Attention(Module):

    def __init__(self, args):
        super(Attention, self).__init__()
        self.score = nn.Linear(2*args.hidden_size, args.hidden_size)

    def forward(self, decoder_outputs, encoder_outputs, source_lengths):
        """
        Return attention scores.
        args:
        decoder_outputs: BxTx*
        encoder_outputs: TxBx*
        returns:
        attention scores: Bx1xT
        """
        projected_encoder_outputs = self.score(encoder_outputs) \
                                        .permute(1, 2,
                                    0) # batch first
        scores = decoder_outputs.bmm(projected_encoder_outputs)
        scores = scores.squeeze(1)
        mask = length_to_mask(source_lengths, source_lengths[0])
        scores.data.masked_fiil_(mask, float('-inf'))
        scores = F.softmax(scores, dim=1)
        return scores.unsqueeze(1)

class Decoder(Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(args.vocab_size, args.embed_size)
        self.rnn = nn.LSTM(input_size = args.embed_size,
                                 hidden_size = 2 * args.hidden_size,
                                 num_layers = 4)
        self.output = nn.Linear(3*args.hidden_size, args.hidden_size)
        self.predict = nn.Linear(args.hidden_size, args.vocab_size)
        self.attention = Attention(args)

    def forward(self, target, encoder_outputs, source_lengths, hidden=None):
        """
        args:
        target: A LongTensor contains a word of the target sentence. size: B*1.
        """
        embed_target = self.embed(target)
        decoder_outputs, decoder_hiddens = self.rnn(embed_target)
        atten_scores = self.attention(decoder_outputs,
                                      encoder_outputs, source_lengths)
        context = atten_scores.bmm(encoder_outputs)
        concat = torch.cat([context, decoder_outputs], -1)
        atten_outputs = F.tanh(self.output(concat))
        predictions = self.predict(atten_outputs)
        return predictions, decoder_hiddens, atten_outputs

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

