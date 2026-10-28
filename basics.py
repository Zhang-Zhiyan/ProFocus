from six.moves import cPickle, range
import pickle
from collections import Counter
import torch
import warnings
import torch.nn as nn

PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'
SPECIAL_SYMBOLS = [PAD, SOS, EOS, UNK]

def pickle_data(file_name, *args):
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def unpickle_data(file_name, python2_to_3=False):
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1))


def iterate_in_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def torch_load_model(checkpoint_file, map_location=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.load(checkpoint_file, map_location=map_location)
    return model

class Vocabulary(object):
    def __init__(self, special_symbols=None):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.intialize_special_symbols()

    def intialize_special_symbols(self):
        self.special_symbols = SPECIAL_SYMBOLS

        for s in self.special_symbols:
            self.add_word(s)

        for s in self.special_symbols:
            name = s.replace('<', '')
            name = name.replace('>', '')
            setattr(self, name, self(s))

    def n_special(self):
        return len(self.special_symbols)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[UNK]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text, max_len=None, add_begin_end=True):
        encoded = [self(token) for token in text]
        if max_len is not None:
            encoded = encoded[:max_len]

        if add_begin_end:
            encoded = [self(SOS)] + encoded + [self(EOS)]

        if max_len is not None:
            encoded += [self(PAD)] * (max_len - len(text))
        return encoded

    def decode(self, tokens):
        return [self.idx2word[token] for token in tokens]

    def decode_print(self, tokens):
        exclude = set([self.word2idx[s] for s in [SOS, EOS, PAD]])
        words = [self.idx2word[token] for token in tokens if token not in exclude]
        return ' '.join(words)

    def __iter__(self):
        return iter(self.word2idx)

    def save(self, file_name):
        with open(file_name, mode="wb") as f:
            pickle.dump(self, f, protocol=2)

    @staticmethod
    def load(file_name):
        with open(file_name, 'rb') as f:
            vocab = pickle.load(f)
        return vocab
