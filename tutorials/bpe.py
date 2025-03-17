# coding=utf-8
import re
from collections import defaultdict


def init_vocabulary(corpus):
    vocab = defaultdict(int)
    charset = set()
    for word in corpus:
        word_with_marker = '_' + word
        chars = list(word_with_marker)
        charset.update(chars)
        tokenized_word = ' '.join(chars)
        vocab[tokenized_word] += 1
    return vocab, charset


def get_pair_counts(vocab):
    pair_counts = defaultdict(int)
    for tokenized_word, cnt in vocab.items():
        # 按空格分隔，得到 tokens
        tokens = tokenized_word.split()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pair_counts[pair] += cnt
    return pair_counts


def merge_pair(vocab, pair):
    new_vocab = {}
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for tokenized_word, cnt in vocab.items():
        # 原来的以空格分隔的两个 token 合并起来
        new_tokenized_word = pattern.sub(''.join(pair), tokenized_word)
        new_vocab[new_tokenized_word] = cnt
    return new_vocab


def byte_pair_encoding(corpus, vocab_size):
    # 初始状态：词列表与字符列表（字符作为最基本的 token）
    vocab, charset = init_vocabulary(corpus)
    # history of merge actions
    merges = []
    tokens = set(charset)
    print(f'original tokens: {len(tokens)}, target size: {vocab_size}')
    # 按频次逐一添加 token
    while len(tokens) < vocab_size:
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            break
        most_freq_pair = max(pair_counts, key=pair_counts.get)
        print('new most_freq_pair:', most_freq_pair, pair_counts[most_freq_pair])
        merges.append(most_freq_pair)
        vocab = merge_pair(vocab, most_freq_pair)
        new_token = ''.join(most_freq_pair)
        tokens.add(new_token)

    # vocab：仍然是在 word 这一角度去看待语料库
    # tokens：用于 tokenizer
    # charset：原始的字符集
    return vocab, merges, charset, tokens


def tokenize_word(word, merges, vocab, charset, unk_token='<unk>'):
    word = '_' + word
    if word in vocab:
        return [word]
    tokens = [ch if ch in charset else unk_token for ch in word]
    for left, right in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i:i+2] == [left, right]:
                tokens[i:i+2] = [left + right]
            else:
                i += 1
    return tokens


if __name__ == '__main__':
    c = """The function generates a vocabulary that represents words as sequences of characters and tracks their counts 
    A more efficient approach initializes the vocabulary with all unique words in the corpus and their counts
    This function processes a corpus to produce the components needed for a tokenizer
    It initializes the vocabulary and character set, creates an empty merges list for storing merge operations
    and sets tokens to the initial character set
    Over time, tokens grows to include all unique tokens the tokenizer will be able to generate
    While actual performance varies by system, the optimized approach consistently delivers better speed
    For languages without spaces, like Chinese, or for multilingual models
    the initial space-based tokenization is typically skipped. Instead, the text is split into individual characters
    We're now ready to examine the core ideas of language modeling
    We'll begin with traditional count-based methods and cover neural network-based techniques in later chapters"""
    c = c.split()
    # voc, cs = init_vocabulary(c.split())
    # print(f'vocab size: {len(voc)}, vocab:', voc)
    # print(f'charset size: {len(cs)}')
    # init_pairs = get_pair_counts(voc)
    # print(f'init pair count: {len(init_pairs)}')
    # print(init_pairs)

    v = byte_pair_encoding(c, 150)
    print(v)

    print(tokenize_word('tokenizer?', v[1], v[0], v[2]))  # ['_token', 'iz', 'er', '<unk>']
