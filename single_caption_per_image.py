import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from basics import iterate_in_chunks,cross_entropy
ALL_METRICS = {'emo_alignment', 'metaphor', 'lcs'}
metaphorical_substrings = {'could be',
                           'appears to be',
                           'appear to be',
                           'reminds me',
                           'remind me',
                           'seems like',
                           'looks like',
                           'look like',
                           'is like',
                           'are like',
                           'think of',
                           'resembles',
                           'resembling'
                           }


@torch.no_grad()
def text_to_emotion(txt2em_clf, encoded_tokens, device, batch_size=1000):
    txt2em_clf.eval()
    emotion_txt_preds = []
    for chunk in iterate_in_chunks(encoded_tokens, batch_size):
        emotion_txt_preds.append(txt2em_clf(chunk.to(device)).cpu())

    emotion_txt_preds = torch.cat(emotion_txt_preds)
    maximizers = torch.argmax(emotion_txt_preds, -1)
    return emotion_txt_preds, maximizers


def occurrence_list_to_distribution(list_of_ints, n_support):
    distribution = np.zeros(n_support, dtype=np.float32)
    for i in list_of_ints:
        distribution[i] += 1
    distribution /= sum(distribution)
    return distribution


def dominant_maximizer(a_list):
    u_elements, u_cnt = np.unique(a_list, return_counts=True)

    has_umax = u_cnt.max() > len(a_list) / 2

    if len(u_cnt) >= 2:
        a, b = sorted(u_cnt)[-2:]
        if a == b:
            has_umax = False

    umax = u_elements[u_cnt.argmax()]
    return has_umax, umax


def emotional_alignment(hypothesis, emotions, vocab, txt2em_clf, device):

    hypothesis_tokenized = hypothesis.apply(lambda x: x.split())
    max_len = hypothesis_tokenized.apply(lambda x: len(x)).max()
    hypothesis = hypothesis_tokenized.apply(lambda x: np.array(vocab.encode(x, max_len=max_len)))
    hypothesis = torch.from_numpy(np.vstack(hypothesis))
    pred_logits, pred_maximizer = text_to_emotion(txt2em_clf, hypothesis, device)

    n_emotions = 9
    emo_dists = torch.from_numpy(np.vstack(emotions.apply(lambda x: occurrence_list_to_distribution(x, n_emotions))))
    x_entropy = cross_entropy(pred_logits, emo_dists).item()

    has_max, maximizer = zip(*emotions.apply(dominant_maximizer))
    emotion_mask = np.array(has_max)
    masked_emotion = np.array(maximizer)[emotion_mask]

    guess_correct = masked_emotion == pred_maximizer[emotion_mask].cpu().numpy()
    accuracy = guess_correct.mean()

    return accuracy, x_entropy


def makes_metaphor_via_substring_matching(sentences, substrings=None):
    if substrings is None:
        substrings = metaphorical_substrings

    makes_metaphor = []
    for s in sentences:
        yes = False
        for m in substrings:
            if m in s:
                yes = True
                break
        makes_metaphor.append(yes)
    return makes_metaphor


def lcs(s1, s2):
    matrix = [[[] for _ in range(len(s2))] for _ in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = [s1[i]]
                else:
                    matrix[i][j] = matrix[i-1][j-1] + [s1[i]]
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1], key=len)
    cs = matrix[-1][-1]
    return cs


def captions_lcs_from_training_utterances(captions_tokenized, train_utters_tokenized):
    maximizers =  np.zeros(len(captions_tokenized), dtype=int)
    max_lcs = np.zeros(len(captions_tokenized))
    averages = np.zeros(len(captions_tokenized))
    for i, caption in enumerate(tqdm(captions_tokenized)):
        caption_res = [len(lcs(caption, tr_example)) for tr_example in train_utters_tokenized]
        max_loc = np.argmax(caption_res)
        maximizers[i] = max_loc
        max_lcs[i] = caption_res[max_loc]
        averages[i] = np.mean(caption_res)
    return max_lcs, averages, maximizers


def apply_basic_evaluations(hypothesis, references, ref_emotions, txt2emo_clf, text2emo_vocab,
                            lcs_sample=None, train_utterances=None, 
                            device="cuda", random_seed=2021,
                            methods_to_do=ALL_METRICS):
    results = []
    stat_track = ['mean', 'std']
    
    if 'emo_alignment' in methods_to_do:
        emo_accuracy, emo_xentopy = emotional_alignment(hypothesis, ref_emotions, text2emo_vocab, txt2emo_clf, device)
        stats = pd.Series(emo_accuracy, dtype=float)
        stats = stats.describe()[stat_track]
        stats = pd.concat([pd.Series({'metric': 'Emo-Alignment-ACC'}), stats])
        results.append(stats)

        stats = pd.Series(emo_xentopy, dtype=float)
        stats = stats.describe()[stat_track]
        stats = pd.concat([pd.Series({'metric': 'Emo-Alignment-XENT'}), stats])
        results.append(stats)
        print('EMO-ALIGN: done')

    if 'metaphor' in methods_to_do:
        met_mask = makes_metaphor_via_substring_matching(hypothesis)
        stats = pd.Series(met_mask, dtype=float)
        stats = stats.describe()[stat_track]
        stats = pd.concat([pd.Series({'metric': 'Metaphors'}), stats])
        results.append(stats)
        print('Metaphor-like expressions: Done')

    if 'lcs' in methods_to_do:
        np.random.seed(random_seed)
        train_utters_tokenized = [u.split() for u in train_utterances]
        uts = pd.Series(train_utters_tokenized).sample(lcs_sample[0]).to_list()
        hypo_token = hypothesis.apply(lambda x: x.split()).sample(lcs_sample[1]).to_list()

        max_lcs, mean_lcs, _ = captions_lcs_from_training_utterances(hypo_token, uts)
        stats = pd.Series(max_lcs).describe()[stat_track]
        stats = pd.concat([pd.Series({'metric': 'max-LCS'}), stats])
        results.append(stats)
        stats = pd.Series(mean_lcs).describe()[stat_track]
        stats = pd.concat([pd.Series({'metric': 'mean-LCS'}), stats])
        results.append(stats)
        print('Novelty via Longest Common Subsequence: Done')

    return results
