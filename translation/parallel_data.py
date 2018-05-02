import torch
import unicodedata
import string
from collections import Counter
import re
import operator
import random
import pandas as pd
from torch import Tensor, LongTensor
from torch.utils.data import Dataset
import sys
sys.path.insert(0, "../")
from seq2seq.data import pad_batch

class Vocab():
    def __init__(self, path, name=""):
        self.name = name
        self.SOS = 0
        self.EOS = 1
        self.PAD = 2
        words = [s.strip() for s in open(path).readlines()]
        self.index2word = dict((i+4, w) for (i,w) in enumerate(words))
        self.index2word[0] = "<start>"
        self.index2word[1] = "<end>"
        self.index2word[2] = "<pad>"
        self.index2word[3] = "<unk>"
        self.word2index = dict((w,i) for (i,w) in self.index2word.items())
        self.vocab_size = len(self.index2word)

    def to_vec(self, text):
        vec = []
        unknown_count = 0
        for w in text.split(' '):
            if w in self.word2index:
                vec.append(self.word2index[w])
            else:
                print(w)
                vec.append(3)
                unknown_count += 1
        return vec, unknown_count

    def to_text(self, vec):
        return " ".join([self.index2word[i] for i in vec])

def normalizeString(s):
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        )
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([',\".!?])", r" \1", s)
    return s

def extract_vocab(lang1, lang2, topk, min_len=3, max_len=10):
    eng_counts = Counter()
    fre_counts = Counter()
    n_words = 0
    lines = open('../data/translation/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    for sentence in lines:
        sentence = normalizeString(sentence)
        eng, fre = sentence.split('\t')
        eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )
        if not eng.startswith(eng_prefixes): continue
        fre, eng = fre.split(' '), eng.split(' ')
        if min_len <= len(fre) <= max_len and min_len <= len(eng) <= max_len:
            eng_counts.update(eng)
            fre_counts.update(fre)
    print("total English words #: {}".format(len(eng_counts)))
    print("total French words #: {}".format(len(fre_counts)))
    eng_frequent = [w for (w, c) in eng_counts.most_common(topk)]
    fre_frequent = [w for (w, c) in fre_counts.most_common(topk)]
    with open('../data/translation/English', 'w') as eng_dict:
        eng_dict.write('\n'.join(eng_frequent))
    with open('../data/translation/French', 'w') as fre_dict:
        fre_dict.write('\n'.join(fre_frequent))

def readLangs(lang1, lang2, reverse=False, mode="train"):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('../Data/translation/%s-%s-%s.txt' % (lang1, lang2, mode), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p, min_len, max_len):
    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )
    return min_len < len(p[0].split(' ')) < max_len and \
        min_len < len(p[1].split(' ')) < max_len and \
        p[0].startswith(eng_prefixes)

class ParallelData(Dataset) :

    def __init__(self, source_vocab, target_vocab, source_path, target_path):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.source, self.source_lens, self.target, self.target_lens = self.__vectorize(source_path, target_path)
        self.length = self.source.size()[0]

    def __vectorize(self, source_path, target_path):
        cols = range(50)
        source_frame = pd.read_csv(source_path, delimiter=" ", names=cols)
        source_lens = (50-source_frame.isnull().sum(axis=1).as_matrix()).tolist()
        source = torch.from_numpy(source_frame.fillna(self.source_vocab.PAD).as_matrix()).long()
        target_frame = pd.read_csv(target_path, delimiter=" ", names=cols)
        target_lens = (50-target_frame.isnull().sum(axis=1)).tolist()
        target = torch.from_numpy(target_frame.fillna(self.target_vocab.PAD).as_matrix()).long()
        return source, source_lens, target, target_lens

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.source[idx], self.source_lens[idx], self.target[idx], self.target_lens[idx]

if __name__ == "__main__":
    English = Vocab("../data/translation/English")
    French = Vocab("../data/translation/French")
    print(English.vocab_size)


