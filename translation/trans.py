import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys
from parallel_data import ParallelData, Vocab
sys.path.insert(0, "../seq2seq")
from utils import parse, length_to_mask, masked_cross_entropy_loss, save_checkpoint
from seq import Encoder, Decoder, run
from tensorboardX import SummaryWriter
from data import pad_batch
import os
import random
from time import strftime, localtime, time

if __name__ == "__main__":
    args = parse()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cuda_prompt = "you are using cuda." if args.cuda else "you are not using cuda."
    print("start model building, "+cuda_prompt)

    print("start data loading: train data at {}, test data at {}".format(args.train_path, args.test_path))
    English = Vocab("../data/translation/English")
    French = Vocab("../data/translation/French")
    train_data = ParallelData(French, English, "../data/translation/French-train-source.txt", "../data/translation/English-train-target.txt")
    test_data = ParallelData(French, English, "../data/translation/French-val-source.txt", "../data/translation/English-val-target.txt")
    collate = lambda x : pad_batch(x, train_data.source_vocab.PAD)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               collate_fn = collate,
                                               num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              collate_fn = collate,
                                              num_workers=args.num_workers)

    print("finish data loading.")
    print("preparing directory {}".format(args.dir))
    os.makedirs(args.dir, exist_ok=True)
    print("building model")
    encoder = Encoder(args, train_data.source_vocab.vocab_size).cuda() if args.cuda else Encoder(args,train_data.source_vocab.vocab_size)
    decoder = Decoder(args, train_data.target_vocab.vocab_size).cuda() if args.cuda else Decoder(args,train_data.target_vocab.vocab_size)

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

    train_loss = 0
    train_correct = 0
    train_total = 0
    count = 0
    time_last = time()
    for epoch in range(args.epochs):
        for batch in enumerate(train_loader):
            batch_id = batch[0]
            if batch_id < start_batch: continue
            #sample_prob = 1 - (1/(len(train_loader)*args.epochs))*batch_id
            sample_prob = 0.5
            correct, total, loss = \
            run(args, train_data.source_vocab, train_data.target_vocab,
                    encoder, decoder, encoder_optim, decoder_optim, batch, writer, "train", sample_prob)
            train_loss += loss
            train_correct += correct
            train_total += total
            count += 1
            if (batch_id) % args.log_interval == 0:
                accu = train_correct/train_total
                train_loss /= count
                time_now = time()
                time_diff = time_now - time_last
                time_last = time_now
                writer.add_scalar('train/accuracy', accu, (epoch * len(train_loader) + batch_id) / args.log_interval)
                writer.add_scalar('train/loss', train_loss, (epoch * len(train_loader) + batch_id) / args.log_interval)
                print(count)
                print("Batch {}: train accuracy: {:.2%}, loss: {}, time use: {:.2}s.".format(batch_id, accu, train_loss, time_diff))
                train_loss = 0
                train_correct = 0
                train_total = 0
                count = 0
            if (batch_id) % args.save_interval == 0:
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
                    correct, total, loss = \
                            run(args, train_data.source_vocab, train_data.target_vocab,
                                    encoder, decoder, encoder_optim, decoder_optim, val_batch, writer, "validate")
                    val_correct += correct
                    val_total += total
                    val_loss += loss
                    run(args, train_data.source_vocab, train_data.target_vocab,
                            encoder, decoder, encoder_optim, decoder_optim, val_batch, writer, "greedy")
                val_loss /= len(test_loader)
                val_accuracy = val_correct / val_total
                print("epoch {} test # {}: test accuracy {:.2%}, test averaged loss {}".format(epoch, batch_id//args.eval_interval, val_accuracy, val_loss))
                writer.add_scalar('val/accuracy', val_accuracy, epoch*len(train_loader)+batch_id)
                writer.add_scalar('val/loss', val_loss, epoch*len(train_loader)+batch_id)
                if args.lr_schedule == "multi":
                    encoder_scheduler.step(val_loss)
                    decoder_scheduler.step(val_loss)
