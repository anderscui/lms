# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import jieba

corpus = [
    '我特别特别喜欢看这部电影。',
    '这部电影真的很好看，强烈推荐，它的名字时步履不停，是枝裕和导演',
    '今天天气真好，是难得的好天气',
    '今天天气是难得的好天气',
    '我今天去电影院看了一部电影，午夜巴黎，伍迪·艾伦',
    '今天电影院的电影很好看，尤其是伍迪·艾伦的午夜巴黎',
]

# jieba.initialize()


def tokenize(sent: str):
    return [token for token in jieba.cut(sent)]


def build_vocab(tokenized: list[list[str]]):
    vocab = {}
    for sent in tokenized:
        for word in sent:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def transform_corpus(tokenized, vocab: dict):
    vectors = []
    for sent in tokenized:
        sent_vec = [0] * len(vocab)
        for token in sent:
            sent_vec[vocab[token]] += 1
        vectors.append(sent_vec)
    return vectors


def cosine_sim(v1, v2):
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)


def calc_corpus_sims(corpus_vecs):
    sims = np.zeros((len(corpus_vecs), len(corpus_vecs)))
    for i in range(len(corpus_vecs)):
        for j in range(len(corpus_vecs)):
            sims[i][j] = cosine_sim(corpus_vecs[i], corpus_vecs[j])
    return sims


corpus_tokenized = [tokenize(sent) for sent in corpus]
corpus_vocab = build_vocab(corpus_tokenized)
corpus_vectors = transform_corpus(corpus_tokenized, corpus_vocab)
corpus_sims = calc_corpus_sims(corpus_vectors)

# for sent in corpus_tokenized:
#     print(sent)

# print(corpus_vocab)
for sent, vec in zip(corpus, corpus_vectors):
    print(sent)
    print(vec)
    print()

print(corpus_sims)

plt.rcParams['font.family'] = ['Lantinghei SC']
plt.rcParams['font.sans-serif'] = ['Lantinghei SC']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots()
cax = ax.matshow(corpus_sims, cmap=plt.cm.Blues)
fig.colorbar(cax)
# ax.set_title('Bag of words')
# fig.set_size_inches(15, 15)

ax.set_xticks(range(len(corpus)))
ax.set_yticks(range(len(corpus)))
ax.set_xticklabels(corpus, rotation=45, ha='left')
ax.set_yticklabels(corpus)
plt.show()
