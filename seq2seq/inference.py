import torch
from seq import Encoder, Decoder, run
from torch import Tensor, LongTensor
from torch.autograd import Variable
from data import  OpenSubVocab, OpenSub, sort_batch
from utils import parse, length_to_mask
from torch.nn.functional import softmax
import os

def greedy(test_loader, encoder, decoder, use_cuda):
    for batch_id ,(source, source_lens, target, target_lens)in enumerate(test_loader):
        source, target = Variable(source, volatile=True), Variable(target, volatile=True)
        if use_cuda: source, target = source.cuda(), target.cuda()
        batch_size = source.size()[1]
        encoder_outputs, encoder_last_hidden = encoder(source, source_lens, None)
        max_target_len = max(target_lens)
        decoder_hidden = encoder_last_hidden
        target_slice = Variable(torch.zeros(batch_size).fill_(test_data.target_vocab.SOS).long(), volatile=True)
        pred_seq = torch.zeros_like(target.data)
        pred_lens = torch.ones(batch_size).byte()
        end = torch.zeros(batch_size).byte()
        if use_cuda:
            source, target = source.cuda(), target.cuda()
            target_slice = target_slice.cuda()
            pred_seq = pred_seq.cuda()
            end = end.cuda()
            pred_lens = pred_lens.cuda()
        for l in range(max_target_len):
            predictions, decoder_hidden, atten_scores = decoder(target_slice,
                                                                encoder_outputs,
                                                                source_lens,
                                                                decoder_hidden)
            pred_words = predictions.data.topk(2, 1)[1]
            unk = pred_words[:, 0] == test_data.target_vocab.UNK
            pred_words = pred_words[:, 0] * (1-unk).long() + pred_words[:, 1] * unk.long()
            pred_seq[l] = pred_words
            target_slice = Variable(pred_words, volatile=True)
            eos = torch.eq(pred_seq[l], test_data.target_vocab.EOS)
            end = (end + eos.cuda())>0
            pred_lens += (end == 0)

        for i in range(batch_size):
            print("Given source sequence:\n {}".format(vocab.to_text(source.data[:source_lens[i], i])))
            print("target sequence is:\n {}".format(vocab.to_text(target.data[:target_lens[i], i])))
            print("greedily decoded sequence is:\n {}".format(vocab.to_text(pred_seq[:pred_lens[i], i])))
        raise NotImplementedError

class Beam():
    def __init__(self, beam_size, vocab, alpha, n_best, use_cuda):
        self.beam_size = beam_size
        self.vocab = vocab
        self.alpha = alpha
        self.n_best = n_best

        self.prevs = [] # pointer to sequence in beam
        self.nexts = [torch.zeros(beam_size).fill_(vocab.SOS)]
        if use_cuda: self.nexts = [t.cuda() for t in self.nexts]
        self.attns = []
        self.scores = torch.zeros(beam_size)
        self.all_scores = []
        if use_cuda: self.scores = self.scores.cuda()
        self.finished = [] # list of tuples, (index within beam, output index, score)
        self.stop = False

    def get_last_words(self):
        return self.nexts[-1]
    
    def get_last_root(self):
        return self.prevs[-1]

    def advance(self, logits, attn):
        """
        Args:
        `logits`: log probability of each candidate sequence for generating next word, beam_size x vocab_size
        `attn`: attention vectors of decoder
        """
        if len(self.prevs) == 0:
            beam_scores = logits[0]
        else:
            beam_scores = self.scores.unsqueeze(1).expand_as(logits) + logits
            for i in range(self.nexts[-1].size(0)):
                if self.nexts[-1][i] == self.vocab.EOS:
                    beam_scores[i] = -1e20 
                
        
        
            #TODO: Block Children of finish sentence
            #TODO: Normalization over length

        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_word_id = flat_beam_scores.topk(self.beam_size, 0, True, True)
        self.all_scores.append(self.scores)
        self.scores = best_scores
        prev = best_word_id / self.vocab.vocab_size
        prev = prev.data.long()
        self.prevs.append(prev)
        next_idx = (best_word_id % self.vocab.vocab_size).data.long()
        self.nexts.append(next_idx)
        self.attns.append(attn.index_select(0, prev))
        for idx, word_idx in enumerate(self.nexts[-1]):
            if word_idx == self.vocab.EOS:
                self.finished.append((idx, len(self.nexts)-1, (self.scores.data[idx])/(len(self.nexts)-1)))
        if self.nexts[-1][0] == self.vocab.EOS:
            self.all_scores.append(self.scores)
            self.stop = True

    def topk(self, k):
        """
        If this beam has finished searching, get the top k best sequence. If there are less than k completed sentences,
        add partial sentences.
        """
        self.finished.sort(key=lambda x : -x[2]) #TODO: Check why this is inverse
        scores = [s for _, _, s in self.finished]
        idx = [(word_idx, beam_idx) for (word_idx, beam_idx, _) in self.finished]
        makeup = k-len(idx)
        for i, (score, word_idx) in enumerate(zip(self.scores, self.nexts[-1])):
            if i > makeup -1: continue
            scores.append(score/(len(self.nexts)-1))
            idx.append((i, len(self.nexts)-1))
                       
        def get_pred(word_idx, beam_idx):
            pred = []
            attn = []
            for i in range(len(self.prevs[:beam_idx]), -1, -1):
                pred.append(self.nexts[i][word_idx])
                attn.append(self.attns[i-1][word_idx])
                word_idx = self.prevs[i-1][word_idx]
            attn.reverse()
            pred.reverse()
            return pred, torch.stack(attn)
        preds = [get_pred(*x) for x in idx]
        sentences, attns = zip(*preds)
        return sentences, attns

def beam_search(encoder, decoder, test_loader, beam_size, vocab, alpha, n_best, max_tgt_len, use_cuda, path):
    for batch_id ,(source, source_lens, target, target_lens)in enumerate(test_loader):
        source, target = Variable(source, volatile=True), Variable(target, volatile=True)
        if use_cuda: source, target = source.cuda(), target.cuda()
        batch_size = source.size()[1]
        encoder_outputs, encoder_last_hidden = encoder(source, source_lens, None)
        decoder_hidden = encoder_last_hidden
        make_beam = lambda : Beam(beam_size, vocab, alpha, n_best, use_cuda)
        beams = [make_beam() for _ in range(batch_size)]
        decoder_hidden = (decoder_hidden[0].repeat(1,beam_size,1), decoder_hidden[1].repeat(1,beam_size,1))
        encoder_outputs = encoder_outputs.repeat(1,beam_size,1)
        source_lens = torch.LongTensor(source_lens).repeat(1,beam_size,1).view(-1).tolist()
        for l in range(max_tgt_len):
            last_words = torch.stack([b.get_last_words() for b in beams])
            last_words = Variable(last_words).t().contiguous().view(1, -1).squeeze(0).long()
            if use_cuda: last_words = last_words.cuda()
            logits, decoder_hidden, atten_scores = decoder(last_words,
                                                                encoder_outputs,
                                                                source_lens,
                                                                decoder_hidden)
            logits = softmax(logits, 1)
            logits = logits.view(beam_size, batch_size, -1)
            atten_scores = atten_scores.view(beam_size, batch_size, -1)
            
            for j, b in enumerate(beams):
                b.advance(logits[:, j], atten_scores.data[:, j])
                last_roots = b.get_last_root()
                for d in decoder_hidden:
                    layer_size = d.size(0)
                    beam_batch = d.size(1)
                    hidden_size = d.size(2)
                    sent_states = d.view(layer_size, beam_size, beam_batch // beam_size,
                            hidden_size)[:, :, j]
                    sent_states.data.copy_(sent_states.data.index_select(1, last_roots))

        for i in range(batch_size):
            print("Given source sequence:\n {}".format(vocab.to_text(source.data[:source_lens[i], i])))
            print("target sequence is:\n {}".format(vocab.to_text(target.data[:target_lens[i], i])))
            pred_seq, _ = beams[i].topk(n_best)
            print("beam search sequence is:\n \
                   1. {} \n \
                   2. {} \n \
                   3. {}".format(vocab.to_text(pred_seq[0]), vocab.to_text(pred_seq[1]), vocab.to_text(pred_seq[2])))
        raise NotImplementedError

if __name__ == "__main__":
    args = parse()
    assert args.resume is not None
    torch.manual_seed(args.seed)
    if args.cuda: torch.cuda.manual_seed(args.seed)

    cuda_prompt = "you are using cuda." if args.cuda else "you are not using cuda."
    print("start model building, "+cuda_prompt)
    vocab = OpenSubVocab(args.vocab_path)
    test_data = OpenSub(args, vocab, args.test_path)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=8,
                                              shuffle=True, collate_fn=sort_batch,
                                              num_workers=args.num_workers)
    print("finish data loading.")
    encoder = Encoder(args, test_data.source_vocab.vocab_size).cuda() if args.cuda else Encoder(args, test_data.source_vocab.vocab_size)
    decoder = Decoder(args, test_data.target_vocab.vocab_size).cuda() if args.cuda else Decoder(args, test_data.target_vocab.vocab_size)
    checkpoint = torch.load(os.path.join(args.dir, args.resume))
    start_batch = checkpoint['batch_id']
    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    beam_search(encoder, decoder, test_loader, args.beam_size, vocab, args.alpha, args.n_best, 20, args.cuda, None)
#     greedy(test_loader, encoder, decoder, args.cuda)
