import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from data import  Vocab, OpenSub, sort_batch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import parse, length_to_mask, masked_cross_entropy_loss, save_checkpoint
from tensorboardX import SummaryWriter
import os
import sys
import random
from time import strftime, localtime, time

class Encoder(Module):

    def __init__(self, args, vocab_size):
        super(Encoder, self).__init__()
        self.dropout_prob = args.dropout
        self.embedding = nn.Embedding(vocab_size, args.embed_size)
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
        #projected_encoder_outputs = self.score(encoder_outputs) \
        #                                .permute(1, 2, 0) # batch first
        projected_encoder_outputs = encoder_outputs.permute(1,2,0)
        decoder_outputs = decoder_outputs.transpose(0,1)
        scores = decoder_outputs.bmm(projected_encoder_outputs)
        scores = scores.squeeze(1)
        mask = length_to_mask(source_lengths, source_lengths[0])
        if scores.is_cuda: mask = mask.cuda()
        scores.data.masked_fill_(1-mask, float('-inf'))
        scores = F.softmax(scores, dim=1)
        return scores.unsqueeze(1)

class Decoder(Module):

    def __init__(self, args, vocab_size):
        super(Decoder, self).__init__()
        self.dropout_prob = args.dropout
        self.embed = nn.Embedding(vocab_size, args.embed_size)
        self.rnn = nn.LSTM(input_size = args.embed_size,
                           hidden_size = 2 * args.hidden_size,
                           num_layers = args.num_layers,
                           dropout = self.dropout_prob,
                           bidirectional=False)
        self.output = nn.Linear(4*args.hidden_size, args.hidden_size)
        self.predict = nn.Linear(args.hidden_size, vocab_size)
        self.attention = Attention(args).cuda() if args.cuda else Attention(args)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.uniform(param, -0.08, 0.08)

    def forward(self, l, target, encoder_outputs, source_lengths, hidden=None):
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


def run(args, source_vocab, target_vocab, encoder, decoder, encoder_optim, decoder_optim, batch, writer, mode, sample_prob=1):
    batch_id, (source, source_lens, target, target_lens) = batch
    if mode == "train":
        encoder.train()
        decoder.train()
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
    elif mode == "validate" or mode == "greedy":
        encoder.eval()
        decoder.eval()
    if args.cuda: source, target = source.cuda(), target.cuda()
    source, target = Variable(source), Variable(target)
    batch_size = source.size()[1]
    encoder_outputs, encoder_last_hidden = encoder(source, source_lens, None)
    max_target_len = max(target_lens)
    decoder_hidden = encoder_last_hidden
    target_slice = Variable(torch.zeros(batch_size).fill_(target_vocab.SOS).long())
    decoder_outputs = Variable(torch.zeros(args.global_max_target_len, batch_size, target_vocab.vocab_size)) # preallocate
    pred_seq = torch.zeros_like(target.data)
    if args.cuda:
        source, target = source.cuda(), target.cuda()
        target_slice = target_slice.cuda()
        decoder_outputs = decoder_outputs.cuda()
        pred_seq = pred_seq.cuda()
    for l in range(max_target_len):
        predictions, decoder_hidden, atten_scores = decoder(l, target_slice, encoder_outputs, source_lens, decoder_hidden)
        decoder_outputs[l] = predictions
        pred_words = predictions.data.max(1)[1]

        pred_seq[l] = pred_words
        if mode == "train" or mode == "validate":
            coin = random.random()
            if coin > sample_prob:
                target_slice = Variable(pred_words).long()
            else:
                target_slice = target[l] # use teacher forcing
        elif mode == "greedy":
            target_slice = Variable(pred_words) # use teacher forcing
        if args.cuda: target_slice = target_slice.cuda()
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

    current_lr = encoder_optim.param_groups[0]['lr']
    if mode == "validate" or mode == "greedy":
        if batch_id == 0 and mode == "greedy":
            i = random.randint(0, batch_size-1)
            print("Given source sequence:\n {}".format(source_vocab.to_text(source.data[:source_lens[i], i])))
            print("target sequence is:\n {}".format(target_vocab.to_text(target.data[:target_lens[i], i])))
            print("greedily decoded sequence is:\n {}".format(target_vocab.to_text(pred_seq[:, i])))
        return correct, total, loss.data[0]
    elif mode == "train":
        if (batch_id+1) % args.log_interval == 0:
            writer.add_scalar('train/lr', current_lr, batch_id)
            i = random.randint(0, batch_size-1)
            print("Given source sequence:\n {}".format(source_vocab.to_text(source.data[:source_lens[i], i])))
            print("target sequence is:\n {}".format(target_vocab.to_text(target.data[:target_lens[i], i])))
            print("teacher forcing generated sequence is:\n {}".format(target_vocab.to_text(pred_seq[:target_lens[i], i])))

        nn.utils.clip_grad_norm(encoder.parameters(), args.clip_thresh)
        nn.utils.clip_grad_norm(decoder.parameters(), args.clip_thresh)
        encoder_optim.step()
        decoder_optim.step()
        return correct, total, loss.data[0]

if __name__ == "__main__":
    args = parse()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cuda_prompt = "you are using cuda." if args.cuda else "you are not using cuda."
    print("start model building, "+cuda_prompt)

    print("start data loading: train data at {}, test data at {}".format(args.train_path, args.test_path))
    vocab = Vocab(args.vocab_path)
    #train_data = CornellMovie(vocab, args.train_path)
    #test_data = CornellMovie(vocab, args.test_path)
    train_data = OpenSub(args, vocab, args.train_path)
    test_data = OpenSub(args, vocab, args.test_path)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True, collate_fn=sort_batch,
                                               num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=True, collate_fn=sort_batch,
                                              num_workers=args.num_workers)

    print("finish data loading.")
    print("preparing directory {}".format(args.dir))
    os.makedirs(args.dir, exist_ok=True)
    print("building model")
    encoder = Encoder(args, 25004).cuda() if args.cuda else Encoder(args, 25004)
    decoder = Decoder(args, 25004).cuda() if args.cuda else Decoder(args, 25004)

    if args.optim == "SGD":
        encoder_optim = optim.SGD(encoder.parameters(), lr=args.lr, momentum=args.momentum)
        decoder_optim = optim.SGD(decoder.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optim == "Adam":
        encoder_optim = optim.Adam(encoder.parameters())
        decoder_optim = optim.Adam(decoder.parameters())
    else:
        raise Exception("Invalid optimizer type {}".format(args.optim))

    if args.lr_schedule == "multi":
        encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optim, 'min', factor=0.5, patience=2, verbose=True, min_lr=0.1)
        decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optim, 'min', factor=0.5, patience=2, verbose=True, min_lr=0.1)

    time_stamp = strftime("%y%m%d-%H:%M", localtime())
    if args.resume is None:
        writer = SummaryWriter(log_dir=os.path.join(".", "runs", "{}-{}".format(args.log_name, time_stamp)))
        with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
            f.write(" ".join(sys.argv))
            f.write("\n")
        start_batch = 0
        start_epoch = 0
    else:
        print('resume training...')
        writer = SummaryWriter(log_dir=os.path.join(".", "runs", "{}".format(args.log_name)))
        checkpoint = torch.load(os.path.join(args.dir, args.resume))
        start_batch = checkpoint['batch_id']
        encoder.load_state_dict(checkpoint['encoder_state'])
        decoder.load_state_dict(checkpoint['decoder_state'])
        encoder_optim.load_state_dict(checkpoint['encoder_opt_state'])
        decoder_optim.load_state_dict(checkpoint['decoder_opt_state'])
    print("start training...\ntotal step #: {}".format(len(train_loader)*args.epochs))
    print("logging per {} batches".format(args.log_interval))
    print("evaluating per {} batches".format(args.eval_interval))
    print("saving per {} batches".format(args.save_interval))
    print("total {} epochs".format(args.epochs))

    train_loss = []
    train_correct = 0
    train_total = 0
    for epoch in range(args.epochs):
        for batch in enumerate(train_loader):
            batch_id = batch[0]
            if batch_id < start_batch: continue
            #sample_prob = 1 - (1/(len(train_loader)*args.epochs))*batch_id
            sample_prob = 1
            correct, total, loss = run(args, train_data.source_vocab, train_data.target_vocab, encoder, decoder, encoder_optim, decoder_optim, batch, "train", sample_prob)
            train_loss += loss
            train_correct += correct
            train_total += total
            if (batch_id) % args.log_interval == 0:
                accu = train_correct/train_total
                avg_loss = sum(train_loss)/len(train_loss)
                time_now = time()
                time_diff = time_now - time_last
                time_last = time_now
                writer.add_scalar('train/accuracy', accu, (epoch * len(train_loader) + batch_id) / args.log_interval)
                writer.add_scalar('train/loss', avg_loss, (epoch * len(train_loader) + batch_id) / args.log_interval)
                print(count)
                print("Batch {}: train accuracy: {:.2%}, loss: {}, time use: {:.2}s.".format(batch_id, accu, avg_loss, time_diff))
                train_loss = 0
                train_correct = 0
                train_total = 0
                count = 0
            if (batch_id+1) % args.save_interval == 0:
                save_checkpoint(
                        args.dir,
                        batch_id,
                        log_name = os.path.join(".", "runs", "{}-{}".format(args.log_name, time_stamp)),
                        encoder_state = encoder.state_dict(),
                        decoder_state = decoder.state_dict(),
                        encoder_opt_state = encoder_optim.state_dict(),
                        decoder_opt_state = decoder_optim.state_dict()
                        )

            if (batch_id+1) % args.eval_interval == 0:
                val_correct = 0
                val_total = 0
                val_loss = 0
                for val_batch in enumerate(test_loader):
                    correct, total, loss = run(args, train_data.source_vocab, train_data.target_vocab, encoder, decoder, encoder_optim, decoder_optim, val_batch, "validate")
                    val_correct += correct
                    val_total += total
                    val_loss += loss
                    run(args, etrain_data.source_vocab, train_data.target_vocab, ncoder, decoder, encoder_optim, decoder_optim, val_batch, "greedy")
                val_loss /= len(test_loader)
                val_accuracy = val_correct / val_total
                print("test # {}: test accuracy {:.2%}, test averaged loss {}".format(batch_id//args.eval_interval, val_accuracy, val_loss))
                writer.add_scalar('val/accuracy', val_accuracy, epoch*len(train_loader)+batch_id)
                writer.add_scalar('val/loss', val_loss, epoch*len(train_loader)+batch_id)
                if args.lr_schedule == "multi":
                    encoder_scheduler.step(val_loss)
                    decoder_scheduler.step(val_loss)
