import torch
from seq import Encoder, Decoder, run
from torch import Tensor, LongTensor
from torch.autograd import Variable
from data import  Vocab, OpenSub, pad_batch
from utils import parse, length_to_mask

def greedy(args, test_loader, encoder, decoder):
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
        if args.cuda:
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


if __name__ == "__main__":
    args = parse()
    assert args.resume is not None
    torch.manual_seed(args.seed)
    if args.cuda: torch.cuda.manual_seed(args.seed)

    cuda_prompt = "you are using cuda." if args.cuda else "you are not using cuda."
    print("start model building, "+cuda_prompt)
    test_data = OpenSub(args, args.test_path)
    vocab = Vocab(args.vocab_path)
    PAD = test_data.PAD
    collate = lambda x:pad_batch(x, PAD)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False, collate_fn=collate,
                                              num_workers=args.num_workers)
    print("finish data loading.")
    encoder = Encoder(args).cuda() if args.cuda else Encoder(args)
    decoder = Decoder(args).cuda() if args.cuda else Decoder(args)
    checkpoint = torch.load(os.path.join(args.dir, args.resume))
    start_batch = checkpoint['batch_id']
    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    greedy(args, test_loader, encoder, decoder)
