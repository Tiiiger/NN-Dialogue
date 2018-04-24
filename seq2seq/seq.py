import torch
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
if args.cuda:
    torch.cuda.manual_seed(args.seed)

time_stamp = strftime("%y%m%d-%H:%M", localtime())
writer = SummaryWriter(log_dir=os.path.join(".", "runs", "{}-{}".format(args.log_name, time_stamp)))
cuda_prompt = "you are using cuda." if args.cuda else "you are not using cuda."
print("start model building, "+cuda_prompt)

class Encoder(Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.dropout_prob = args.dropout
        self.embedding = nn.Embedding(args.vocab_size+4, args.embed_size)
        # Only accept 4 layers bi-directional LSTM right now
        self.rnn = nn.LSTM(input_size=args.embed_size,
                                  hidden_size=args.hidden_size,
                                  num_layers=args.num_layers,
                                  dropout = self.dropout_prob,
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
        if args.cuda: mask = mask.cuda()
        scores.data.masked_fill_(1-mask, float('-inf'))
        scores = F.softmax(scores, dim=1)
        return scores.unsqueeze(1)

class Decoder(Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.dropout_prob = args.dropout
        self.embed = nn.Embedding(args.vocab_size+4, args.embed_size)
        self.rnn = nn.LSTM(input_size = args.embed_size,
                           hidden_size = 2 * args.hidden_size,
                           num_layers = args.num_layers,
                           dropout = self.dropout_prob,
                           bidirectional=False)
        self.output = nn.Linear(4*args.hidden_size, args.hidden_size)
        self.predict = nn.Linear(args.hidden_size, args.vocab_size+4)
        self.attention = Attention(args).cuda() if args.cuda else Attention(args)
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
        atten_scores = self.attention(decoder_outputs, encoder_outputs, source_lengths)
        context = atten_scores.bmm(encoder_outputs.transpose(0,1))
        concat = torch.cat([context, decoder_outputs.transpose(0,1)], -1)
        atten_outputs = F.tanh(self.output(concat))
        predictions = self.predict(atten_outputs)
        predictions = predictions.squeeze(1)
        return predictions, decoder_hiddens, atten_scores

print("start data loading: train data at {}, test data at {}".format(args.train_path, args.test_path))
train_data = OpenSub(args, args.train_path)
test_data = OpenSub(args, args.test_path)
vocab = Vocab(args.vocab_path)
PAD = train_data.PAD
collate = lambda x:pad_batch(x, PAD)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=args.batch_size,
                                           shuffle=False, collate_fn=collate,
                                           num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=args.batch_size,
                                          shuffle=True, collate_fn=collate,
                                          num_workers=args.num_workers)

print("finish data loading.")
print("preparing directory {}".format(args.dir))
os.makedirs(args.dir, exist_ok=True)
print("building model")
encoder = Encoder(args).cuda() if args.cuda else Encoder(args)
decoder = Decoder(args).cuda() if args.cuda else Decoder(args)


encoder_optim = optim.SGD(encoder.parameters(), lr=args.lr, momentum=args.momentum)
decoder_optim = optim.SGD(decoder.parameters(), lr=args.lr, momentum=args.momentum)
if args.lr_schedule == "multi":
    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optim, 'min', factor=0.5, patience=2, verbose=True, min_lr=0.1)
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optim, 'min', factor=0.5, patience=2, verbose=True, min_lr=0.1)

if args.resume is None:
    with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
        f.write(" ".join(sys.argv))
        f.write("\n")
    start_batch = 0
    start_epoch = 0
else:
    print('resume training...')
    checkpoint = torch.load(os.path.join(args.dir, args.resume))
    start_batch = checkpoint['batch_id']
    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    encoder_optim.load_state_dict(checkpoint['encoder_opt_state'])
    decoder_optim.load_state_dict(checkpoint['decoder_opt_state'])

def run(batch_id, source, source_lens, target, target_lens, mode):
    if mode == "train":
        encoder.train()
        decoder.train()
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
    elif mode == "validate":
        encoder.eval()
        decoder.eval()
    time_start = time()
    if args.cuda: source, target = source.cuda(), target.cuda()
    source, target = Variable(source), Variable(target)
    batch_size = source.size()[1]
    encoder_outputs, encoder_last_hidden = encoder(source, source_lens, None)
    max_target_len = max(target_lens)
    decoder_hidden = encoder_last_hidden
    target_slice = Variable(torch.zeros(batch_size).fill_(train_data.SOS).long())
    decoder_outputs = Variable(torch.zeros(args.global_max_target_len, batch_size, args.vocab_size+4)) # preallocate
    pred_seq = torch.zeros_like(target.data)
    if args.cuda:
        source, target = source.cuda(), target.cuda()
        target_slice = target_slice.cuda()
        decoder_outputs = decoder_outputs.cuda()
        pred_seq = pred_seq.cuda()
    for l in range(max_target_len):
        predictions, decoder_hidden, atten_scores = decoder(target_slice, encoder_outputs, source_lens, decoder_hidden)
        decoder_outputs[l] = predictions
        pred_words = predictions.data.max(1)[1]
        pred_seq[l] = pred_words
        target_slice = target[l] # use teacher forcing
        # detach hidden states
        for h in decoder_hidden:
            h.detach_()
    mask = Variable(length_to_mask(target_lens)).transpose(0,1).float()
    if args.cuda: mask = mask.cuda()

    loss = masked_cross_entropy_loss(decoder_outputs[:max_target_len], target, mask)
    if mode == "train": loss.backward()

    correct = torch.eq(target.data.float() , pred_seq.float()) * mask.data.byte()
    correct = correct.float().sum()
    total = mask.data.float().sum()
    accuracy = correct / total

    time_now = time()
    time_diff = time_now - time_start
    current_lr = encoder_optim.param_groups[0]['lr']
    if mode == "validate":
        return correct, total, loss.data[0]
    elif mode == "train":
        if batch_id % args.log_interval == 0:
            writer.add_scalar('train/accuracy', accuracy, batch_id)
            writer.add_scalar('train/loss', loss, batch_id)
            writer.add_scalar('train/lr', current_lr, batch_id)
            i = random.randint(1, batch_size)
            print("Given source sequence:\n {}".format(vocab.to_text(source.data[:source_lens[i], i])))
            print("target sequence is:\n {}".format(vocab.to_text(target.data[:target_lens[i], i])))
            print("generated sequence is:\n {}".format(vocab.to_text(pred_seq[:target_lens[i], i])))
            print("Batch {}: train accuracy: {:.2%}, loss: {}, lr: {}, time use: {:.2}s.".format(batch_id, accuracy, loss.data[0], current_lr, time_diff))

        if batch_id % args.save_interval == 0:
            save_checkpoint(
                    args.dir,
                    batch_id,
                    encoder_state = encoder.state_dict(),
                    decoder_state = decoder.state_dict(),
                    encoder_opt_state = encoder_optim.state_dict(),
                    decoder_opt_state = decoder_optim.state_dict()
                    )

        nn.utils.clip_grad_norm(encoder.parameters(), args.clip_thresh)
        nn.utils.clip_grad_norm(decoder.parameters(), args.clip_thresh)
        encoder_optim.step()
        decoder_optim.step()
        return correct, total, loss.data[0]

print("start training...\ntotal batch #: {}".format(len(train_loader)))
print("logging per {} batches".format(args.log_interval))
print("evaluating per {} batches".format(args.eval_interval))
print("saving per {} batches".format(args.save_interval))
print("total {} epochs".format(args.epochs))

for epoch in range(args.epochs):
    for batch_id ,(source, source_lens, target, target_lens)in enumerate(train_loader):
        if batch_id < start_batch: continue
        correct, total, loss = run(batch_id,source, source_lens, target, target_lens, "train")
        if (batch_id+1) % args.eval_interval == 0:
            val_correct = 0
            val_total = 0
            val_loss = 0
            for batch_id, (source, source_lens, target, target_lens)in enumerate(test_loader):
                correct, total, loss = run(batch_id, source, source_lens, target, target_lens, "validate")
                val_correct += correct
                val_total += total
                val_loss += loss
            val_loss /= len(test_loader)
            val_accuracy = val_correct / val_total
            print("test # {}: test accuracy {:.2%}, test averaged loss {}".format(batch_id//args.eval_interval, val_accuracy, val_loss))
            writer.add_scalar('val/accuracy', test_accuracy, epoch*len(train_loader)+batch_id)
            writer.add_scalar('val/loss', test_loss/len(test_loader), epoch*len(train_loader)+batch_id)
            if args.lr_schedule == "multi":
                encoder_scheduler.step(val_loss)
                decoder_scheduler.step(val_loss)

def evaluate():
    """
    TODO:
 1. get BLEU score
    2. save attention
    """
    raise NotImplementedError
