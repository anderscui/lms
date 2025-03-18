# coding=utf-8
import os
import pickle
import re
import tarfile
import time
import urllib.request
from collections import defaultdict

START_WORD_MARKER = '_'


def download_file(url, filename):
    """
    Downloads a file from a URL if it doesn't exist locally.
    Prevents redundant downloads by checking file existence.

    Args:
        url (str): URL to download the file from
        filename (str): Local path to save the downloaded file

    Returns:
        None: Prints status messages about download progress
    """
    # Check if file already exists to avoid re-downloading
    if not os.path.exists(filename):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, filename)
        print("Download completed.")
    else:
        print(f"{filename} already downloaded.")


def is_within_directory(directory, target):
    """
    Security check to prevent path traversal attacks by verifying target path.
    Ensures extracted files remain within the intended directory.

    Args:
        directory (str): Base directory path to check against
        target (str): Target path to validate

    Returns:
        bool: True if target is within directory, False otherwise
    """
    # Convert both paths to absolute form for comparison
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    # Get common prefix to check containment
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory


def safe_extract_tar(tar_file, required_files):
    """
    Safely extracts specific files from a tar archive with security checks.
    Prevents path traversal attacks and extracts only required files.

    Args:
        tar_file (str): Path to the tar archive file
        required_files (list): List of filenames to extract

    Returns:
        None: Extracts files and prints progress

    Raises:
        Exception: If path traversal attempt is detected
    """
    with tarfile.open(tar_file, "r:gz") as tar:
        # Perform security check on all archive members
        for member in tar.getmembers():
            if not is_within_directory('.', member.name):
                raise Exception("Attempted Path Traversal in Tar File")

        # Extract only the specified files
        for member in tar.getmembers():
            if any(member.name.endswith(file) for file in required_files):
                # Remove path prefix for safety
                member.name = os.path.basename(member.name)
                tar.extract(member, '.')
                print(f"Extracted {member.name}")


def create_word_generator(filepath):
    """
    Creates a generator that yields words from a text file one at a time.
    Memory efficient way to process large text files.

    Args:
        filepath (str): Path to text file to read

    Returns:
        generator: Yields individual words from the file
    """
    def generator():
        with open(filepath, 'r') as f:
            for line in f:
                for word in line.split():
                    yield word
    return generator()


def download_and_prepare_data(url):
    """
    Downloads, extracts, and prepares dataset for training.
    Handles both downloading and extraction with security checks.

    Args:
        url (str): URL of the dataset to download

    Returns:
        generator: Word generator for the training data
    """
    required_files = ["train.txt", "test.txt"]
    filename = os.path.basename(url)

    # Download dataset if needed
    download_file(url, filename)

    # Extract required files if they don't exist
    if not all(os.path.exists(file) for file in required_files):
        print("Extracting files...")
        safe_extract_tar(filename, required_files)
        print("Extraction completed.")
    else:
        print("'train.txt' and 'test.txt' already extracted.")

    # Create and return word generator
    return create_word_generator("train.txt")


def init_vocabulary(corpus):
    """
    Creates initial vocabulary from corpus by splitting words into characters.
    Adds word boundary marker `START_WORD_MARKER` and tracks unique characters.

    Args:
        corpus (iterable): Iterator or list of words to process

    Returns:
        tuple: (vocabulary dict mapping tokenized words to counts,
               set of unique characters in corpus)
    """
    vocab = defaultdict(int)
    charset = set()

    for word in corpus:
        word_with_marker = START_WORD_MARKER + word
        chars = list(word_with_marker)
        charset.update(chars)
        tokenized_word = ' '.join(chars)
        vocab[tokenized_word] += 1
    return vocab, charset


def get_pair_counts(vocab):
    """
    Counts frequencies of adjacent symbol pairs in the vocabulary.
    Used to identify most common pairs for merging.

    Args:
        vocabulary (dict): Dictionary mapping tokenized words to their counts

    Returns:
        defaultdict: Maps token pairs to their frequency counts
    """
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


def tokenize_word(word, merges, charset, unk_token='<UNK>'):
    word = START_WORD_MARKER + word
    # if word in vocab:
    #     return [word]
    tokens = [ch if ch in charset else unk_token for ch in word]
    # 这里的 merges 是“学习到”的内容，其结果可能会与想象的不同
    # 比如 a b c d -> a b cd -> ab cd，而不是 abc d。
    for left, right in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i:i+2] == [left, right]:
                tokens[i:i+2] = [left + right]
            else:
                i += 1
    return tokens


def build_merge_map(merges):
    """
    Creates a mapping from token pairs to their merged forms.
    Preserves merge order for consistent tokenization.

    Args:
        merges (list): List of merge operations

    Returns:
        dict: Maps token pairs to (merged_token, merge_priority) tuples
    """
    merge_map = {}
    # Build map with merge priorities
    for i, (left, right) in enumerate(merges):
        merged_token = left + right
        merge_map[(left, right)] = (merged_token, i)
    return merge_map


def tokenize_word_fast(word, merge_map, vocabulary, charset, unk_token="<UNK>"):
    """
    Optimized tokenization function using pre-computed merge map.
    Produces identical results to original algorithm but faster.

    Args:
        word (str): Word to tokenize
        merge_map (dict): Mapping of token pairs to merged forms
        vocabulary (dict): Current vocabulary dictionary
        charset (set): Set of known characters
        unk_token (str): Token to use for unknown characters

    Returns:
        list: List of tokens for the word
    """
    # Check if word exists in vocabulary as-is
    word_with_prefix = START_WORD_MARKER + word
    if word_with_prefix in vocabulary:
        print(f'exists in vocab: {word_with_prefix}')
        return [word_with_prefix]

    # Initialize with characters, replacing unknown ones
    tokens = [char if char in charset else unk_token for char in word_with_prefix]

    # Keep merging until no more merges possible
    while True:
        # Find all possible merge operations
        pairs_with_positions = []
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in merge_map:
                merged_token, merge_priority = merge_map[pair]
                pairs_with_positions.append((i, pair, merged_token, merge_priority))

        # Exit if no more merges possible
        if not pairs_with_positions:
            break

        # Sort by merge priority and position for consistency
        pairs_with_positions.sort(key=lambda x: (x[3], x[0]))

        # Apply first valid merge
        pos, pair, merged_token, _ = pairs_with_positions[0]
        tokens[pos:pos+2] = [merged_token]

    return tokens


def save_tokenizer(vocab, merges, charset, tokens, filename="tokenizer.pkl"):
    """
    Saves tokenizer state to a pickle file for later use.

    Args:
        merges (list): List of merge operations
        charset (set): Set of known characters
        tokens (set): Set of all tokens
        filename (str): Path to save tokenizer state

    Returns:
        None: Saves tokenizer to disk
    """
    with open(filename, "wb") as f:
        pickle.dump({
            "vocab": vocab,
            "merges": merges,
            "charset": charset,
            "tokens": tokens
        }, f)

def load_tokenizer(filename="tokenizer.pkl"):
    """
    Loads tokenizer state from a pickle file.

    Args:
        filename (str): Path to saved tokenizer state

    Returns:
        dict: Dictionary containing tokenizer components
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def train_tokenizer():
    # Configuration parameters
    vocab_size = 10_000  # Target vocabulary size
    max_corpus_size = 5_000_000  # Maximum number of words to process
    data_url = "https://www.thelmbook.com/data/news"  # Dataset source

    # Download and prepare training data
    word_gen = download_and_prepare_data(data_url)

    # Collect corpus up to maximum size
    word_list = []
    for word in word_gen:
        word_list.append(word)
        if len(word_list) >= max_corpus_size:
            break
    print(f'{len(word_list)} words loaded')

    # Train BPE tokenizer
    print("Training BPE tokenizer...")
    start_time = time.time()
    vocab, merges, charset, tokens = byte_pair_encoding(word_list, vocab_size)
    elapsed = time.time() - start_time
    print(f'Time elapsed for training BPE tokenizer: {elapsed}')

    # Save trained tokenizer
    print("Saving the tokenizer...")
    save_tokenizer(vocab, merges, charset, tokens)


def test_tokenizer():
    print("Loading the tokenizer...")
    tokenizer = load_tokenizer()

    # Tokenize the sample sentence using the loaded tokenizer
    sentence = "Let's proceed to the language modeling part."

    start_time = time.time()
    tokenized_sentence = [tokenize_word(word, tokenizer["merges"], tokenizer["charset"]) for word in sentence.split()]
    elapsed = time.time() - start_time
    print("\nSentence tokenized with the straightforward implementation:")
    for word, tokens in zip(sentence.split(), tokenized_sentence):
        print(f"{word} -> {tokens}")
    print("--- Elapsed: %s seconds ---" % (elapsed))

    merge_map = build_merge_map(tokenizer["merges"])
    start_time = time.time()
    fast_tokenized_sentence = [tokenize_word_fast(word, merge_map, tokenizer['vocab'], tokenizer["charset"]) for word in sentence.split()]
    elapsed = time.time() - start_time
    print("\nSentence tokenized with a fast implementation:")
    for word, tokens in zip(sentence.split(), fast_tokenized_sentence):
        print(f"{word} -> {tokens}")
    print("--- Elapsed: %s seconds ---" % (time.time() - start_time))

    print("\nVocabulary size:", len(tokenizer["tokens"]))


if __name__ == '__main__':
    # c = """The function generates a vocabulary that represents words as sequences of characters and tracks their counts
    # A more efficient approach initializes the vocabulary with all unique words in the corpus and their counts
    # This function processes a corpus to produce the components needed for a tokenizer
    # It initializes the vocabulary and character set, creates an empty merges list for storing merge operations
    # and sets tokens to the initial character set
    # Over time, tokens grows to include all unique tokens the tokenizer will be able to generate
    # While actual performance varies by system, the optimized approach consistently delivers better speed
    # For languages without spaces, like Chinese, or for multilingual models
    # the initial space-based tokenization is typically skipped. Instead, the text is split into individual characters
    # We're now ready to examine the core ideas of language modeling
    # We'll begin with traditional count-based methods and cover neural network-based techniques in later chapters"""
    # c = c.split()
    # # voc, cs = init_vocabulary(c.split())
    # # print(f'vocab size: {len(voc)}, vocab:', voc)
    # # print(f'charset size: {len(cs)}')
    # # init_pairs = get_pair_counts(voc)
    # # print(f'init pair count: {len(init_pairs)}')
    # # print(init_pairs)
    #
    # v = byte_pair_encoding(c, 150)
    # print(v)
    #
    # print(tokenize_word('tokenizer?', v[1], v[0], v[2]))  # ['_token', 'iz', 'er', '<UNK>']

    # train_tokenizer()
    test_tokenizer()
