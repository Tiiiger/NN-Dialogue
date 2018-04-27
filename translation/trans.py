import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import parse, length_to_mask, masked_cross_entropy_loss, save_checkpoint
from tensorboardX import SummaryWriter
import os
import sys
import random
from time import strftime, localtime, time

from data import ParallelData

if __name__ == "__main__":
    args = parse()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cuda_prompt = "you are using cuda." if args.cuda else "you are not using cuda."
    print("start model building, "+cuda_prompt)

    print("start data loading: train data at {}, test data at {}".format(args.train_path, args.test_path))
    train_data = ParallelData()
    test_data = ParallelData()
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

    if args.optim == "SGD":
        encoder_optim = optim.SGD(encoder.parameters(), lr=args.lr, momentum=args.momentum)
        decoder_optim = optim.SGD(decoder.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optim == "Adam":
        encoder_optim = optim.Adam(encoder.parameters())
        decoder_optim = optim.Adam(decoder.parameters())
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

    for epoch in range(args.epochs):
        for batch_id ,(source, source_lens, target, target_lens)in enumerate(train_loader):
            if batch_id < start_batch: continue
            sample_prob = 1 - (1/(len(train_loader)*args.epochs))*batch_id
            correct, total, loss = run(args, encoder, decoder, encoder_optim, decoder_optim, batch_id,source, source_lens, target, target_lens, "train", sample_prob)
            if (batch_id+1) % args.eval_interval == 0:
                val_correct = 0
                val_total = 0
                val_loss = 0
                for val_batch_id, (source, source_lens, target, target_lens)in enumerate(test_loader):
                    correct, total, loss = run(args, encoder, decoder, encoder_optim, decoder_optim, val_batch_id, source, source_lens, target, target_lens, "validate")
                    val_correct += correct
                    val_total += total
                    val_loss += loss
                    run(args, encoder, decoder, encoder_optim, decoder_optim, val_batch_id, source, source_lens, target, target_lens, "greedy")
                val_loss /= len(test_loader)
                val_accuracy = val_correct / val_total
                print("test # {}: test accuracy {:.2%}, test averaged loss {}".format(batch_id//args.eval_interval, val_accuracy, val_loss))
                writer.add_scalar('val/accuracy', val_accuracy, epoch*len(train_loader)+batch_id)
                writer.add_scalar('val/loss', val_loss, epoch*len(train_loader)+batch_id)
                if args.lr_schedule == "multi":
                    encoder_scheduler.step(val_loss)
                    decoder_scheduler.step(val_loss)
