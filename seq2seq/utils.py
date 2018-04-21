import argparse
import torch
from torch.nn.functional import softmax, log_softmax
from torch import LongTensor,Tensor
from torch.autograd import Variable

def parse():
    parser = argparse.ArgumentParser(description='Pass Parameters for Seq2Seq Model')
    parser.add_argument('--batch', dest='batch_size', type=int, default=64,
                        help='batch size of training ')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers ')
    parser.add_argument('--pretrain', dest='pretrain', action="store_const",
                        const=True, default=False, help='use the pretrain model \
                        to run')
    parser.add_argument('--trainpath', dest="train_path", type=str,
                        default="../data/processed/t_given_s_dialogue_length2_3_train.txt")
    parser.add_argument('--testpath', dest="test_path", type=str,
                        default="../data/processed/t_given_s_dialogue_length2_3_test.txt")
    parser.add_argument('--vocabsize', dest="vocab_size",
                        type=int, default=25000)
    parser.add_argument('--hiddensize', dest="hidden_size",
                        type=int, default=1000)
    parser.add_argument('--embedsize', dest="embed_size",
                        type=int, default=1000)
    parser.add_argument('--layers', dest="num_layers",
                        type=int, default=4)
    parser.add_argument('--clip', dest="clip_thresh", type=int, default=1)
    parser.add_argument('--dictpath', dest="dict_path",
                        type=str, default="../Data/movie_25000")
    parser.add_argument('--reverse', dest="reverse", action="store_const",
                        const=True, default=False, )
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
    parser.add_argument('--log-name', type=str, default="seq", metavar='S',
                    help='name of current model')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='intervals of writing tensorboard')
    parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                    help='intervals of validating')
    parser.add_argument('--global-max-target-len', type=int, default=20, metavar='N',
                    help='intervals of validating')
    # parser.add_argument('--testpath', dest="test_path", type=str, default="../Data/t_given_s_train.txt")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def length_to_mask(lengths, longest=None):
    if longest == None:
        longest = max(lengths)
    batch_size = len(lengths)
    index = torch.arange(longest).long()
    index = index.expand(batch_size, longest)
    lengths = LongTensor(lengths).unsqueeze(1).expand_as(index)
    mask = index < lengths
    return mask

def masked_cross_entropy_loss(logits, target, mask):
    # credits: https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
    logits_flat = logits.view(-1, logits.size(-1))
    logits_flat = log_softmax(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(logits_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask
    loss = losses.sum()/mask.sum()
    return loss

def get_bleu(predictions, target, length):
    raise NotImplementedError
