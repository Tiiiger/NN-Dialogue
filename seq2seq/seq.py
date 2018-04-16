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
from tensorboardX import SummaryWriter
import os
from time import strftime, localtime

args = parse()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

time_stamp = strftime("%y%m%d-%H:%M", localtime())
writer = SummaryWriter(log_dir=os.path.join("..", "runs", "{}-{}".format(args.log_name, time_stamp)))

class Encoder(Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size+4, args.embed_size)
        # Only accept 4 layers bi-directional LSTM right now
        self.rnn = nn.LSTM(input_size=args.embed_size,
                                  hidden_size=args.hidden_size,
                                  num_layers=4,
                                  bidirectional=True)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.uniform(param, -0.08, 0.08)

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
        self.predict = nn.Linear(args.hidden_size, args.vocab_size+4)
        self.attention = Attention(args)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.uniform(param, -0.08, 0.08)

    def forward(self, target, encoder_outputs, source_lengths, hidden=None):
        """
        args:
        target: A LongTensor contains a word of the target sentence. size: 1*B.
        """
        target = target.unsqueeze(0)
        embed_target = self.embed(target)
        decoder_outputs, decoder_hiddens = self.rnn(embed_target, hidden)
        atten_scores = self.attention(decoder_outputs,
                                      encoder_outputs, source_lengths)
        context = atten_scores.bmm(encoder_outputs.transpose(0,1))
        concat = torch.cat([context, decoder_outputs.transpose(0,1)], -1)
        atten_outputs = F.tanh(self.output(concat))
        predictions = self.predict(atten_outputs)
        predictions = predictions.squeeze(1)
        return predictions, decoder_hiddens, atten_scores

train_data = OpenSub(args)
test_data = OpenSub(args, "../Data/t_given_s_test.txt")
PAD = train_data.PAD
collate = lambda x:pad_batch(x, PAD)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True, collate_fn=collate,
                                           num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=1000,
                                          shuffle=True, collate_fn=collate,
                                          num_workers=args.num_workers)

encoder = Encoder(args).cuda() if args.cuda else Encoder(args)
decoder = Decoder(args).cuda() if args.cuda else Decoder(args)

encoder_optim = optim.SGD(encoder.parameters(), lr=args.lr)
decoder_optim = optim.SGD(decoder.parameters(), lr=args.lr)

def train(epoch):
    encoder.train()
    decoder.train()
    for batch_id, (source, source_lens, target, target_lens) in enumerate(train_loader):
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        source, target = Variable(source), Variable(target)
        if args.cuda:
            source, target = source.cuda(), target.cuda()
        encoder_outputs, encoder_last_hidden = encoder(source, source_lens, None)
        max_target_len = max(target_lens)
        decoder_hidden = encoder_last_hidden
        target_slice = Variable(torch.zeros(args.batch_size).fill_(train_data.SOS).long())
        # preallocate 
        decoder_outputs = Variable(torch.zeros(max_target_len, args.batch_size, args.vocab_size+4)) # preallocate
        pred_seq = torch.zeros_like(target)
        for l in range(max_target_len):
            predictions, decoder_hidden, atten_scores = decoder(target_slice, encoder_outputs, source_lens, decoder_hidden)
            decoder_outputs[l] = predictions
            pred_words = predictions.max(1)[1]
            pred_seq[l] = pred_words
            target_slice = target[l] # use teacher forcing
            #TODO: check if we need to detach
            # detach hidden states
            # for h in decoder_hidden:
            #     h.detach_()
        mask = Variable(length_to_mask(target_lens).float())
        if args.cuda: mask.cuda()

        loss = masked_cross_entropy_loss(decoder_outputs, target, mask)
        loss.backward()

        mask.transpose_(0,1)
        correct = float(torch.eq(target.float() * mask, pred_seq.float() * mask).sum())
        total = float(mask.sum())
        accuracy = correct / total
        print(accuracy)

        if batch_id+1 % args.log_interval == 0:
            step = epoch * len(train_loader) + batch_id
            writer.add_scalar('train/accuracy', accuracy, batch_id)
            writer.add_scalar('train/loss', loss, batch_id)
        #TODO: Gradient Clip
        encoder_optim.step()
        decoder_optim.step()

def test(epoch):
    encoder.eval()
    decoder.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    for batch_id, (source, source_lens, target, target_lens) in enumerate(test_loader):
        source, target = Variable(source, volatile=True), Variable(target, volatile=True)
        if args.cuda: source, target = source.cuda(), target.cuda()
        encoder_outputs, encoder_last_hidden = encoder(source, source_lens, None)
        max_target_len = max(target_lens)
        decoder_hidden = encoder_last_hidden
        target_slice = Variable(torch.zeros(args.batch_size).fill_(train_data.SOS).long())
        decoder_outputs = Variable(torch.zeros(max_target_len, args.batch_size, args.vocab_size+4)) # preallocate
        pred_seq = torch.zeros_like(target)
        for l in range(max_target_len):
            predictions, decoder_hidden, atten_scores = decoder(target_slice, encoder_outputs, source_lens, decoder_hidden)
            decoder_outputs[l] = predictions
            pred_words = predictions.max(1)[1]
            pred_seq[l] = pred_words
            target_slice = pred_words # use own predictions
        mask = Variable(length_to_mask(target_lens).float(), volatile=True)
        if args.cuda: mask.cuda()
        test_loss += masked_cross_entropy_loss(decoder_outputs, target, mask)
        mask.transpose_(0,1)
        test_correct += float(torch.eq(target.float() * mask, pred_seq.float() * mask).sum())
        test_total += float(mask.sum())

    test_loss /= len(test_loader)
    test_accuracy = test_correct / test_total
    writer.add_scalar('val/accuracy', accuracy, epoch)
    writer.add_scalar('val/loss', loss, epoch)

    raise NotImplementedError

def evaluate():
    """
    TODO:
    1. get BLEU score
    2. save attention
    """
    raise NotImplementedError

for ep in range(args.epoch):
    train(ep)
    if ep % args.eval_interval == 0:
        test(ep)
