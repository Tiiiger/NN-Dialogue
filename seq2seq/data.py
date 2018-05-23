import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch import LongTensor, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
import numpy as np

class OpenSubVocab():
    def __init__(self, path):
        self.EOS = 25001
        self.SOS = 25002
        self.PAD = 25003
        self.UNK = 1
        self.index2word = [s.strip() for s in open(path).readlines()]
        self.word2index = dict((w, i) for (i, w) in enumerate(self.index2word))
        self.vocab_size = 25004

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
                sentence += self.index2word[n-1] + " "
        return sentence
    
    def to_vec(self, s):
        return LongTensor([self.word2index[w] for w in s.split()])

class CornellVocab():
    def __init__(self, path, name=""):
        self.name = name
        self.SOS = 0
        self.EOS = 1
        self.PAD = 2
        self.UNK = 3
        words = [s.strip() for s in open(path).readlines()]
        self.index2word = dict((i+4, w) for (i,w) in enumerate(words))
        self.index2word[0] = "<start>"
        self.index2word[1] = "<end>"
        self.index2word[2] = "<pad>"
        self.index2word[3] = "<unk>"
        self.word2index = dict((w,i+1) for (i,w) in self.index2word.items())
        self.vocab_size = len(self.index2word)

    def to_vec(self, text):
        vec = []
        unknown_count = 0
        for w in text.split(' '):
            if w in self.word2index:
                vec.append(self.word2index[w])
            else:
                vec.append(3)
                unknown_count += 1
        return vec, unknown_count

    def to_text(self, vec):
        return " ".join([self.index2word[i] for i in vec])

class OpenSub(Dataset):
    def __init__(self, params, vocab, path=None):
        if path==None:
            path = params.train_path
        self.params = params
        self.source_vocab = vocab
        self.target_vocab = vocab
        self.__read_data(path, params.reverse)
        self.length = self.source.size()[0]

    def __len__(self):
        return self.length

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
        self.source = torch.from_numpy(source_frame.fillna(self.source_vocab.PAD).as_matrix()).long()
        target_frame = pd.read_csv(target_path, delimiter=" ", names=cols)
        self.target_lens = (20-target_frame.isnull().sum(axis=1)).tolist()
        self.target = torch.from_numpy(target_frame.fillna(self.target_vocab.PAD).as_matrix()).long()

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

def sort_batch(batch):
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
    target_words = target_words[:, sort_source]
    target_lens = [target_lens[i] for i in sort_source]

    return source_words, source_lens, target_words, target_lens



class CornellMovie(Dataset) :

    def __init__(self, vocab, path):
        source_path, target_path = path + "_source.txt", path + "_target.txt"
        self.source_vocab = vocab
        self.target_vocab = vocab
        self.source, self.source_lens, self.target, self.target_lens = self.__vectorize(source_path, target_path)
        self.length = self.source.size()[0]

    def __vectorize(self, source_path, target_path):
        cols = range(21)
        source_frame = pd.read_csv(source_path, delimiter=" ", names=cols)
        source_lens = (21-source_frame.isnull().sum(axis=1).as_matrix()).tolist()
        source = torch.from_numpy(source_frame.fillna(self.source_vocab.PAD).as_matrix()).long()
        target_frame = pd.read_csv(target_path, delimiter=" ", names=cols)
        target_lens = (21-target_frame.isnull().sum(axis=1)).tolist()
        target = torch.from_numpy(target_frame.fillna(self.target_vocab.PAD).as_matrix()).long()
        return source, source_lens, target, target_lens

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.source[idx], self.source_lens[idx], self.target[idx], self.target_lens[idx]
