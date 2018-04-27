import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch import LongTensor, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
import numpy as np

class Vocab():
    def __init__(self, path):
        self.EOS = 25001
        self.SOS = 25002
        self.dict = [s.strip() for s in open(path).readlines()]

    def to_text(self, t):
        sentence = ""
        for n in t:
            if n == 25001:
                sentence += "<end>"
            elif n == 25002:
                sentence += "<start>"
            elif n == 25003:
                continue
            else:
                sentence += self.dict[n-1] + " "
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
        self.__read_data(path, params.reverse)
        self.length = self.source.size()[0]

    def __len__(self):
        return self.length

    def __split_to_tensor(self, line):
        line = line.split()
        arr = [int(i) for i in line]
        arr.append(self.EOS)
        return Tensor(arr)

    def __read_data(self, path, reverse):
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
        source_path = path+ "_source.txt"
        target_path = path+ "_target.txt"
        if reverse: source_path, target_path = target_path, source_path
        cols = range(20)
        source_frame = pd.read_csv(source_path, delimiter=" ", names=cols)
        self.source_lens = (20-source_frame.isnull().sum(axis=1).as_matrix()).tolist()
        self.source = torch.from_numpy(source_frame.fillna(self.PAD).as_matrix()).long()
        target_frame = pd.read_csv(target_path, delimiter=" ", names=cols)
        self.target_lens = (20-target_frame.isnull().sum(axis=1)).tolist()
        self.target = torch.from_numpy(target_frame.fillna(self.PAD).as_matrix()).long()
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
        return self.source[idx], self.source_lens[idx], self.target[idx], self.target_lens[idx]

def pad_batch(batch, PAD):
    """
     Pad a batch to have same length for all sequences.
     Args:
         batch: A python list of [batch_size] tuple of tensors representing
                source, source lengths, target, and target lengths sequences
     Returns:
         pad_source: A T*B Variable, B is [batch_size], T is the maximum
                     length of source sequences in the batch.
         pad_source: A T*B Variable, B is [batch_size], T is the maximum
                     length of target sequences in the batch.
         source_lengths: A [1*batch_size] list; lengths[i] is the length of the
                  batch[i].
         target_lengths: A [1*batch_size] list; lengths[i] is the length of the
                  batch[i].
    """
    source, source_lens, target, target_lens = zip(*batch)
    sort_source = np.flipud(np.argsort(source_lens)).tolist()
    max_source_len = source_lens[sort_source[0]]
    source_words = torch.stack(source, 1)[0:max_source_len, :]
    source_words = source_words[:, sort_source]
    source_lens = [source_lens[i] for i in sort_source]
    max_target_len = max(target_lens)
    target_lens = target_lens
    target_words = torch.stack(target, 1)[0:max_target_len, :]

    return source_words, source_lens, target_words, target_lens

