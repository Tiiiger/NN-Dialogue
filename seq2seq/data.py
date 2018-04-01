import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

def parse():
    parser = argparse.ArgumentParser(description='Pass Parameters for Seq2Seq Model')
    parser.add_argument('--batch', dest='batch_size', type=int, default=4,
                        help='define the batch size of training process')
    parser.add_argument('--pretrain', dest='pretrain', action="store_const",
                        const=True, default=False, help='use the pretrain model \
                        to run')
    parser.add_argument('--trainpath', dest="train_path", type=str,
                        default="../Data/t_given_s_train.txt")
    parser.add_argument('--testpath', dest="test_path", type=str,
                        default="../Data/t_given_s_test.txt")
    parser.add_argument('--vocabsize', dest="vocab_size",
                        type=int, default=25000)
    parser.add_argument('--clip', dest="clip_thresh", type=int, default=5)
    parser.add_argument('--dictpath', dest="dict_path",
                        type=str, default="../Data/movie_25000")
    parser.add_argument('--reverse', dest="reverse", action="store_const",
                        const=True, default=False, )
    # parser.add_argument('--testpath', dest="test_path", type=str, default="../Data/t_given_s_train.txt")

    return parser.parse_args()

class Loader(Dataset):
    def __init__(self, params, path=None):
        if path==None:
            path = params.train_path
        self.params = params
        self.EOS = params.vocab_size+1
        self.SOS = params.vocab_size+2
        self.UNKNOWN = params.vocab_size+3
        self.PAD = params.vocab_size+4
        self.last = 0
        self.batch_size = params.batch_size
        self.source, self.target = self.__read_data(path, params.reverse)

    def __split_to_tensor(self, line):
        line = line.split()
        arr = [int(i) for i in line]
        return arr

    def __read_data(self, path):
        """
        Read the data into this data loader. The source sequences if reverse if
        [params.reverse].
        Args:
            path: A String specifying the path to the data file
        Returns:
            source: A python list of tensors; each element is a tensor
                    representing a source sequence. tensors may have different
                    length.
            target: A python list of tensors; each element is a tensor
                    representing a target sequence. tensors may have different
                    length.
        """
        with open(path, "r") as data_file:
            lines = data_file.readlines()
            source = []
            target = []
            for l in lines:
                s, t = l.split('|')
                s = self.__split_to_tensor(s)
                t = self.__split_to_tensor(t)
                source.append(s)
                target.append(t)
            return source, target

    def __getitem__(self, idx):
        """
        Get a batch of source and target sequences.
        Returns:
        source_words: A B*T Variable, B is [batch_size], T is the maximum
                       length of source sequences in the batch.
        source_lengths: A [1*batch_size] tensor; lengths[i] is the length of the
                       batch[i].
        target_words: A B*T Variable, B is [batch_size], T is the maximum
                       length of target sequences in the batch.
        target_lengths: A [1*batch_size] tensor; lengths[i] is the length of the
                       batch[i].
        """
        return self.source[idx], self.target[idx]


def pad_batch(self, batch):
    """
     Pad a batch to have same length for all sequences.
     Args:
         batch: A python list of [batch_size] tuple of tensors representing
                source and target sequences
     Returns:
         pad_source: A B*T Variable, B is [batch_size], T is the maximum
                     length of source sequences in the batch.
         pad_source: A B*T Variable, B is [batch_size], T is the maximum
                     length of target sequences in the batch.
         source_lengths: A [1*batch_size] list; lengths[i] is the length of the
                  batch[i].
         target_lengths: A [1*batch_size] list; lengths[i] is the length of the
                  batch[i].
    """
    pair_sort = sorted(batch, key=lambda p:len(p[0]), reverse=True)
    source, target = zip(*pair_sort)
    max_source_length = len(source[0])
    source_lengths = [len(i) for i in source]
    target_lengths = [len(i) for i in target]
    max_target_length = max(target_lengths)
    pad_source = []
    pad_target = []
    for i in range(len(source)):
        pad_source.append(source[i] + [self.PAD for j in range(max_source_length-source_lengths[i])])
        pad_target.append(target[i] + [self.PAD for j in range(max_target_length-target_lengths[i])])
    source_words = Variable(torch.LongTensor(pad_source)).transpose(0, 1)
    target_words = Variable(torch.LongTensor(pad_target)).transpose(0, 1)
    return source_words, source_lengths, target_words, target_lengths

