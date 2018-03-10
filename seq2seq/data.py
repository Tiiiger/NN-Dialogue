import argparse
from torch import Tensor

def parse():
    parser = argparse.ArgumentParser(description='Pass Parameters for Seq2Seq Model')
    parser.add_argument('--batch', dest='batch_size', type=int,
                        default=4, help='define the batch size of training process')
    parser.add_argument('--pretrain', dest='pretrain' action="store_const", const=True, default=False, 
                        help='use the pretrain model to run')
    parser.add_argument('--trainpath', dest="train_path", type=str, default="../Data/t_given_s_train.txt")
    parser.add_argument('--testpath', dest="test_path", type=str, default="../Data/t_given_s_test.txt")
    parser.add_argument('--vocabsize', dest="vocab_size", type=int, default=25000)
    parser.add_argument('--clip', dest="clip_thresh", type=int, default=5)
    parser.add_argument('--dictpath', dest="dict_path", type=str, default="../Data/movie_25000")
    parser.add_argument('--reverse', dest="reverse", action="store_const", const=True, default=False,
                        type=str, default="../Data/movie_25000")
    # parser.add_argument('--testpath', dest="test_path", type=str, default="../Data/t_given_s_train.txt")

    return parser.parse_args()

class Loader:
    def __init__(self, params):
        self.EOS = params.vocab_size+1
        self.SOS = params.vocab_size+2
        self.UNKNOWN = params.vocab_size+3
        self.last = 0
        self.batch_size = params.batch_size
        self.train_source, self.train_target = self.__read_data(params.reverse)

    def split_to_tensor(line, rev=False):
        line = line.split()
        arr = [int(i) for i in line]
        if rev: arr.reverse()
        return Tensor(arr)

    def __read_data(path=params.train_path):
        """
        Read the data into this data loader. The source sequences if reverse if
        [params.reverse].
        Args:
            path: A String specifying the path to the data file
        Returns:
            source: A python list of tensors; each element is a tensor
                    representing a source sequence.
            target: A python list of tensors; each element is a tensor
                    representing a target sequence.
        """
        raise NotImplementedError

    def __iter__():
        return self

    def __next__():
        max_size = self.data.source()[0]-1
        if self.last * self.batch_size > self.data.source()[0]:
            self.last = 0
            raise StopIteration
        return self.get_batch()

    def __pad_batch(batch):
        """
        Pad a batch to have same length for all sequences.
        Args:
            batch: A python list of [batch_size] tensors representing sequences
        Returns:
            pad_batch: A B*T Variable, B is [batch_size], T is the maximum
                       length of sequences in the batch.
            lengths:   A [1*batch_size] tensor; lengths[i] is the length of the
                       batch[i].
        """
        raise NotImplementedError

    def shuffle(mode="train"):
        raise NotImplementedError

    def get_batch():
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




