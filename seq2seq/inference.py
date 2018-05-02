import torch
from seq import Encoder, Decoder, run
from torch import Tensor, LongTensor
from torch.autograd import Variable
from data import  OpenSubVocab, OpenSub, sort_batch
from utils import parse, length_to_mask

def greedy(test_loader, encoder, decoder, use_cuda):
    for batch_id ,(source, source_lens, target, target_lens)in enumerate(test_loader):
        source, target = Variable(source, volatile=True), Variable(target, volatile=True)
        batch_size = source.size()[1]
        encoder_outputs, encoder_last_hidden = encoder(source, source_lens, None)
        max_target_len = max(target_lens)
        decoder_hidden = encoder_last_hidden
        target_slice = Variable(torch.zeros(batch_size).fill_(test_data.SOS).long(), volatile=True)
        pred_seq = torch.zeros_like(target.data)
        pred_lens = torch.ones(batch_size)
        end = torch.zeros(batch_size)
        if use_cuda:
            source, target = source.cuda(), target.cuda()
            target_slice = target_slice.cuda()
            pred_seq = pred_seq.cuda()
        for l in range(max_target_len):
            predictions, decoder_hidden, atten_scores = decoder(target_slice,
                                                                encoder_outputs,
                                                                source_lens,
                                                                decoder_hidden)
            pred_words = predictions.data.max(1)[1]
            pred_seq[l] = pred_words
            target_slice = Variable(pred_words, volatile=True)
            eos = torch.eq(pred_seq, test_data.EOS)
            end = (end + eos)>0
            pred_lens += (end == 0)

        for i in range(batch_size):
            print("Given source sequence:\n {}".format(vocab.to_text(source.data[:source_lens[i], i])))
            print("target sequence is:\n {}".format(vocab.to_text(target.data[:target_lens[i], i])))
            print("greedily decoded sequence is:\n {}".format(vocab.to_text(pred_seq[:pred_lens[i], i])))


class Sequence():
    def __init__(self, words, hidden, logprob, attention):
        self.words = words
        self.hidden = hidden
        self.logprob = logprob
        self.attention = attention

class Beam():
    def __init__(self, beam_size, vocab, alpha, n_best, use_cuda):
        self.T = torch.cuda if use_cuda else torch
        self.beam_size = beam_size
        self.vocab = vocab
        self.alpha = alpha
        self.n_best = n_best

        self.prevs = [] # pointer to sequence in beam
        self.nexts = [self.T.zeros(beam_size).fill_(vocab.SOS)]
        self.attns = []
        self.scores = self.T.zeros(beam_size)
        self.finished = [] # list of tuples, (index within beam, output index, score)
        self.step = 0
        self.stop = False

    def get_last_words():
        return self.nexts[-1]

    def advance(self, logits, attn):
        """
        Args:
        `logits`: log probability of each candidate sequence for generating next word, beam_size x vocab_size
        `attn`: attention vectors of decoder
        """
        if len(prevs) == 0:
            beam_scores = logits[0]
        else:
            beam_scores = self.scores.expand_as(logits) + logits
            #TODO: Block Children of finish sentence
            #TODO: Normalization over length

        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_word_id = flat_beam_scores.topk(self.beam_size, 0, True, True)
        self.all_scores = best_scores
        prev = best_word_id / self.vocab.vocab_size
        self.prevs.append(prev)
        self.nexts.append(best_word_id % self.vocab.vocab_size)
        self.attns.append(attn.index_select(0, prev))
        for idx, word_idx in enumerate(self.nexts[-1]):
            if word_idx == vocab.EOS:
                self.finished.append(idx, len(self.nexts)-1, scores[idx])
        if self.nexts[-1][0] == vocab.EOS:
            self.all_scores.append(scores)
            self.stop = True
        step += 1

    def topk(self, k):
        """
        If this beam has finished searching, get the top k best sequence. If there are less than k completed sentences,
        add partial sentences.
        """
        self.finished.sort(key=lambda x : -x[2]) #TODO: Check why this is inverse
        scores = [s for _, _, s in self.finished]
        idx = [(word_idx, beam_idx) for (word_idx, beam_idx, _) in self.finished]
        def get_pred(beam_idx, word_idx):
            pred = []
            attn = []
            for i in range(len(self.prevs[:beam_idx], -1, -1)):
                pred.append(self.prevs[i+1][word_idx])
                attn.append(self.attns[i][word_idx])
                word_idx = prevs[i][word_idx]
            return pred.reverse, torch.stack(attn.reverse)
        preds = [get_pred(*x) for x in idx]
        sentences, attns = zip(*preds)
        return sentences, attns



def beam_search(encoder, decoder, test_loader, beam_size, vocab, alpha, n_best, max_tgt_len, use_cuda, path):
    for batch_id ,(source, source_lens, target, target_lens)in enumerate(test_loader):
        source, label = Variable(source, volatile=True), Variable(target, volatile=True)
        batch_size = source.size()[1]
        encoder_outputs, encoder_last_hidden = encoder(source, source_lens, None)
        decoder_hidden = encoder_last_hidden
        make_beam = lambda : Beam(beam_size, vocab, alpha, n_best, use_cuda)
        beams = [make_beam() for _ in range(batch_size)]
        if use_cuda:
            source, target = source.cuda(), target.cuda()
            target_slice = target_slice.cuda()
        for l in range(max_tgt_len):
            last_words = torch.stack([b.get_last_words() for b in beams])
            last_words = Variable(last_words).contiguous().view(1, -1)
            if use_cuda: last_words = last_words.cuda()
            logits, decoder_hidden, atten_scores = decoder(last_words,
                                                                encoder_outputs,
                                                                source_lens,
                                                                decoder_hidden)
            target_slice = Variable(pred_words, volatile=True)
            pred_lens += (end == 0)

        for i in range(batch_size):
            print("Given source sequence:\n {}".format(vocab.to_text(source.data[:source_lens[i], i])))
            print("target sequence is:\n {}".format(vocab.to_text(target.data[:target_lens[i], i])))
            print("greedily decoded sequence is:\n {}".format(vocab.to_text(pred_seq[:pred_lens[i], i])))

    pass

if __name__ == "__main__":
    # args = parse()
    # assert args.resume is not None
    # torch.manual_seed(args.seed)
    # if args.cuda: torch.cuda.manual_seed(args.seed)

    # cuda_prompt = "you are using cuda." if args.cuda else "you are not using cuda."
    # print("start model building, "+cuda_prompt)
    # test_data = OpenSub(args, args.test_path)
    # vocab = Vocab(args.vocab_path)
    # PAD = test_data.PAD
    # collate = lambda x:pad_batch(x, PAD)
    # test_loader = torch.utils.data.DataLoader(test_data,
    #                                           batch_size=args.batch_size,
    #                                           shuffle=False, collate_fn=collate,
    #                                           num_workers=args.num_workers)
    # print("finish data loading.")
    # encoder = Encoder(args).cuda() if args.cuda else Encoder(args)
    # decoder = Decoder(args).cuda() if args.cuda else Decoder(args)
    # checkpoint = torch.load(os.path.join(args.dir, args.resume))
    # start_batch = checkpoint['batch_id']
    # encoder.load_state_dict(checkpoint['encoder_state'])
    # decoder.load_state_dict(checkpoint['decoder_state'])
    # greedy(test_loader, encoder, decoder, args.cuda)
    vocab = OpenSubVocab("../data/movie_25000")
    beam = Beam(3, vocab, 0, 3, False)
    pass
