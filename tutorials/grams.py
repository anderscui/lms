# coding=utf-8
from collections import Counter, defaultdict

book_corpus = [
    '我喜欢吃苹果',
    '我喜欢吃葡萄',
    '我喜欢吃香蕉',
    '我不喜欢吃橘子',
    '我不喜欢吃桃子',
    '她喜欢吃葡萄',
    '她喜欢吃葡萄干',
    '他不喜欢吃香蕉',
    '他喜欢吃苹果',
    '他喜欢吃苹果派',
    '她喜欢吃草莓',
    '她喜欢吃番茄',
    '他喜欢吃番茄酱',
    '他喜欢吃番薯',
    '编辑工作者',
    '编一个故事吧',
    '编程工具集锦',
    '编程爱好者',
    '编程爱好挺酷的不是吗',
    '我喜欢编程序',
    '我喜欢编程式',
    '我喜欢编故事',
    '我喜欢编造谎言',
]


def tokenize(text: str):
    return [c for c in text]


def count_ngrams(corpus: list[str], n=2):
    ngrams_counts = defaultdict(Counter)
    for text in corpus:
        tokens = tokenize(text)
        for i in range(len(tokens) - n + 1):
            ngrams = tuple(tokens[i:i+n])
            prefix = ngrams[:-1]
            token = ngrams[-1]
            ngrams_counts[prefix][token] += 1
    return ngrams_counts


def ngram_probs(ngram_counts: defaultdict):
    gram_probs = defaultdict(Counter)
    for gram_prefix, token_count in ngram_counts.items():
        total = token_count.total()
        for token, cnt in token_count.items():
            gram_probs[gram_prefix][token] = cnt / total
    return gram_probs


def generate_next_token(prefix, gram_probs: defaultdict):
    # print('prefix for gen:', prefix)
    if prefix not in gram_probs:
        return None

    next_tokens = gram_probs[prefix]
    # print('next:', next_tokens)
    if not next_tokens:
        return None
    return next_tokens.most_common(1)[0][0]


def generate_text(prompt: str, gram_probs: defaultdict, n_grams=2, max_len=6):
    prefix_len = n_grams - 1
    cur_tokens = tokenize(prompt)
    for _ in range(max_len - len(prompt)):
        gen_token = generate_next_token(tuple(cur_tokens[-prefix_len:]), gram_probs)
        if gen_token is None:
            break
        cur_tokens.append(gen_token)
    return ''.join(cur_tokens)


N_GRAMS = 5
bigrams_counts = count_ngrams(book_corpus, n=N_GRAMS)
bigrams_probs = ngram_probs(bigrams_counts)
print('bigrams:')
for prefix, count in bigrams_counts.items():
    print('{}'.format(''.join(prefix)), end=': ')
    # print('freq:', dict[count])
    print('prob:', bigrams_probs[prefix])
    print()

# print(generate_next_token(('我',), bigrams_probs))
# print(generate_next_token(('她',), bigrams_probs))
# print(generate_next_token(('他',), bigrams_probs))
print(generate_text('我喜欢吃', bigrams_probs, max_len=6, n_grams=N_GRAMS))
print(generate_text('我喜欢编', bigrams_probs, max_len=15, n_grams=N_GRAMS))
print(generate_text('她喜欢吃葡', bigrams_probs, max_len=15, n_grams=N_GRAMS))
