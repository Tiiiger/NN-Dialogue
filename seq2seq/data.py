import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch import LongTensor, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Vocab():
    def __init__(self, path="../Data/movie_25000"):
        self.D = {}
        with open(path, "r") as f:
            line_num = 1
            for line in f.readlines():
                self.D[line_num] = line.strip()
                line_num += 1

    def to_text(self, t):
        sentence = ""
        for n in t:
            if n == 25002 or n == 25001: continue
            sentence += self.D[n] + " "
        return sentence

class OpenSub(Dataset):
    def __init__(self, params, path=None):
        if path==None:
            path = params.train_path
        self.params = params
        self.EOS = params.vocab_size+1
        self.SOS = params.vocab_size+2
        self.PAD = params.vocab_size+3
        self.vocab_size = params.vocab_size+3
        self.source, self.target = self.__read_data(path)
        self.length = len(self.source)

    def __len__(self):
        return self.length

    def __split_to_tensor(self, line):
        line = line.split()
        arr = [int(i) for i in line]
        arr.append(self.EOS)
        return Tensor(arr)

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
            line_num = len(lines)
            source = []
            target = []
            count = 0
            for l in lines:
                percent = count / line_num
                count += 1
                if percent % 0.1 == 0: print("loading {:%} data".format(percent))
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


def pad_batch(batch, PAD):
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
    batch_size = len(source)
    max_source_length = len(source[0])
    source_lengths = [len(i) for i in source]
    target_lengths = [len(i) for i in target]
    max_target_length = max(target_lengths)
    def pad(tensors, max_len, lens, batch_size):
        pad = Tensor(max_len, batch_size)
        for i in range(batch_size):
            pad[:, i] = torch.cat((tensors[i], PAD*torch.ones(max_len-lens[i])))
        return pad.long()
    source_words = pad(source, max_source_length, source_lengths, batch_size)
    target_words = pad(target, max_target_length, target_lengths, batch_size)
    return source_words, source_lengths, target_words, target_lengths

