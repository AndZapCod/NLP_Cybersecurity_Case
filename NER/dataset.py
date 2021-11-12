import pickle
import torch
from torch.utils.data import Dataset

START_OF_SEQUENCE = '<SOS>'
END_OF_SEQUENCE = '<EOS>'


class ToPaddedTensor:

    def __init__(self, length, pad_idx):
        self.length = length
        self.pad_idx = pad_idx

        with open('data/dics.pkl', 'rb') as f:
            self.token2idx = pickle.load(f)
            self.idx2token = pickle.load(f)
            self.tag2idx = pickle.load(f)
            self.idx2tag = pickle.load(f)

        start_idx = len(self.token2idx)
        end_idx = len(self.token2idx) + 1
        self.token2idx[START_OF_SEQUENCE] = start_idx
        self.token2idx[END_OF_SEQUENCE] = end_idx
        self.idx2token.append(START_OF_SEQUENCE)
        self.idx2token.append(END_OF_SEQUENCE)

        start_idx = len(self.tag2idx)
        end_idx = len(self.tag2idx) + 1
        self.tag2idx[START_OF_SEQUENCE] = start_idx
        self.tag2idx[END_OF_SEQUENCE] = end_idx
        self.idx2tag.append(START_OF_SEQUENCE)
        self.idx2tag.append(END_OF_SEQUENCE)

    def words2idxs(self, tokens_list):
        return [self.token2idx[word] for word in tokens_list]

    def tags2idxs(self, tags_list):
        return [self.tag2idx[tag] for tag in tags_list]

    def padd_sequence(self, seq):
        size = seq.shape[0]
        if size > self.length:
            raise ValueError('Sequence length is greater than padding length.')

        padding = torch.Tensor([self.pad_idx] * (self.length - size)).int()
        seq = torch.cat((seq, padding))
        return seq

    def __call__(self, tok_tag):
        tok, tag = tok_tag

        tok = [START_OF_SEQUENCE] + tok + [END_OF_SEQUENCE]
        tag = [START_OF_SEQUENCE] + tag + [END_OF_SEQUENCE]

        tok = torch.Tensor(self.words2idxs(tok)).long()
        tag = torch.Tensor(self.tags2idxs(tag)).long()
        tok = self.padd_sequence(tok)
        tag = self.padd_sequence(tag)
        return tok, tag

    def idxs2words(self, idxs):
        return [self.idx2token[idx] for idx in idxs]

    def idxs2tags(self, idxs):
        return [self.idx2tag[idx] for idx in idxs]

    def from_padded_tensor(self, tok_tag):
        tok, tag = tok_tag
        tok = self.idxs2words(tok)
        tag = self.idxs2tags(tag)
        return tok, tag


class Corpus(Dataset):

    def __init__(self, file_dir, transform=None):
        with open(file_dir, 'rb') as f:
            tok_tag = pickle.load(f)
            self.tok = tok_tag['tokens']
            self.tag = tok_tag['tags']
            self.n_samples = tok_tag.shape[0]
            self.transform = transform
            self.max_sent_len = max(map(lambda x: len(x), self.tok))

    def __getitem__(self, item):
        tok_tag = self.tok[item], self.tag[item]
        if self.transform:
            tok_tag = self.transform(tok_tag)
        return tok_tag

    def __len__(self):
        return self.n_samples
