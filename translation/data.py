import torch
import unicodedata
import string
import re
import random
from torch import Tensor, LongTensor
from torch.utils.data import Dataset
import sys
sys.path.insert(0, "../")
from seq2seq.data import pad_batch

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False, mode="train"):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('../Data/%s-%s-%s.txt' % (lang1, lang2, mode), encoding='utf-8').\
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
        p[1].startswith(eng_prefixes)

def filterPairs(pairs, min_len, max_len):
    return [pair for pair in pairs if filterPair(pair, min_len, max_len)]

def prepareData(lang1, lang2, reverse=False, mode):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, mode)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, 3, 10)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

class ParallelData(Dataset) :

    def __init__(self, mode="train"):
        assert mode in ["train"; "val"]
        self.SOS = 0
        self.EOS = 1
        self.PAD = 2
        input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
        self.length = len(pairs)
        self.source, self.target = self.vectorize(input_lang, output_lang, pairs)

    def vectorize(self, input_lang, output_lang, pairs):
        def sentence2vec(sen, lang):
            sen_vect  = [lang.word2index[w] for w in sen.split(' ')]
            sen_vect.append(self.EOS)
            return sen_vect
        source = [sentence2vec(pair[0], input_lang) for pair in pairs]
        target = [sentence2vec(pair[1], output_lang).append(self.EOS) for pair in pairs]
        return source, target

    def __len__(self):
        return self.length

    def __getiem__(self, idx):
        return self.source[idx], len(self.source[idx]), self.target[idx], len(self.target[idx])

if __name__ == "__main__":
    train_data = ParallelData()
    collate = lambda x:pad_batch(x, PAD)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=1,
                                               shuffle=False)
    for batch in enumerate(train_loader):
        source, source_lens, target, target_lens = batch
        print(source, source_lens)
        raise NotImplementedError
