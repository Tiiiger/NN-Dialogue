# Credits: https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/evaluate.py
import torch
from collections import Counter
def ngram_precision(prediction, reference, n):
    """
    predictions: T list
    reference: Txn list
    """
    p_ngrams = [tuple(prediction[i:i+n]) for i in range(len(prediction+1-n))]
    r_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference+1-n))]
    p_ngrams_counts = Counter(p_ngrams)
    r_ngrams_counts = Counter(r_ngrams)
    overlap = p_ngrams_counts & r_ngrams_counts
    overlap_count = sum(overlap.values())
    total_count = len(p_ngrams_counts)
    return overlap_count, total_count

def compute_bleu(hypothesis, hypo_lengths, reference, refer_lengths):
    """
    Compute statistics for BLEU.
    Args:
    hypothesis: T*B, T is the longest length, B is batch size
    """
    c = torch.sum(hypo_lengths)
    r = torch.sum(reference_lengths)
    batch_size = size(hypothesis)[1]
    stats = torch.zeros(batch_size, 8)
    for b in range(batch_size):
        for n in range(4):
            overlap, total = ngram_precision(hypothesis[0:hypo_lengths[b], b],
                                             reference[0:refer_lengths[b], b],
                                             n+1)
            stats[b, n] = overlap
            stats[b, n+1] = total
    stats = torch.sum(stats, 0)
    stats = stats.view(4,2)
    precision = stats[:, 0]/stats[:, 1]
    precision = precision.log().sum()
    base = min(0, 1-r/c)
    return math.exp(base+precision)
