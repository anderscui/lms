# coding=utf-8
import re


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.token2ids = vocab
        self.id2tokens = {tid: token for token, tid in vocab.items()}

        self.unk_token = '<|unk|>'
        self.endoftext_token = '<|endoftext|>'

    def tokenize(self, t: str):
        parts = re.split(r'([,.:;?_!"()\']|--|\s)', t)
        parts = [token.strip() for token in parts if token.strip()]
        return parts

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.token2ids.get(token, self.token2ids[self.unk_token]) for token in tokens]

    def decode(self, ids):
        text = " ".join([self.id2tokens[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
