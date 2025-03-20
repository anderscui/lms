# coding=utf-8
import gzip
import io
import math
import os
import pickle
import re
from collections import defaultdict
import random

import requests


def set_seed(seed):
    """
    Sets random seeds for reproducibility.

    Args:
        seed (int): Seed value for the random number generator
    """
    random.seed(seed)


def get_hyperparameters():
    """
    Returns model hyperparameters.

    Returns:
        int: Size of n-grams to use in the model
    """
    n = 5
    return n


def download_corpus(url):
    """
    Downloads and decompresses a gzipped corpus file from the given URL.

    Args:
        url (str): URL of the gzipped corpus file

    Returns:
        str: Decoded text content of the corpus

    Raises:
        HTTPError: If the download fails
    """
    print(f"Downloading corpus from {url}...")
    response = requests.get(url)
    response.raise_for_status()  # Raises an exception for bad HTTP responses

    print("Decompressing and reading the corpus...")
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
        corpus = f.read().decode('utf-8')

    print(f"Corpus size: {len(corpus)} characters")
    return corpus


def download_and_prepare_data(data_url):
    """
    Downloads and prepares training and test data.

    Args:
        data_url (str): URL of the corpus to download

    Returns:
        tuple: (training_tokens, test_tokens) split 90/10
    """
    # Download and extract the corpus
    corpus = download_corpus(data_url)

    # Convert text to tokens
    tokens = tokenize(corpus)

    # Split into training (90%) and test (10%) sets
    split_index = int(len(tokens) * 0.9)
    train_corpus = tokens[:split_index]
    test_corpus = tokens[split_index:]

    return train_corpus, test_corpus


def save_model(model, model_name):
    """
    Saves the trained language model to disk.

    Args:
        model (CountLanguageModel): Trained model to save
        model_name (str): Name to use for the saved model file

    Returns:
        str: Path to the saved model file

    Raises:
        IOError: If there's an error writing to disk
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Construct file path
    model_path = os.path.join('models', f'{model_name}.pkl')

    try:
        print(f"Saving model to {model_path}...")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'n': model.n,
                'ngram_counts': model.ngram_counts,
                'total_unigrams': model.total_unigrams
            }, f)
        print("Model saved successfully.")
        return model_path
    except IOError as e:
        print(f"Error saving model: {e}")
        raise

def load_model(model_name):
    """
    Loads a trained language model from disk.

    Args:
        model_name (str): Name of the model to load

    Returns:
        CountLanguageModel: Loaded model instance

    Raises:
        FileNotFoundError: If the model file doesn't exist
        IOError: If there's an error reading the file
    """
    model_path = os.path.join('models', f'{model_name}.pkl')

    try:
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Create new model instance
        model = CountLanguageModel(model_data['n'])

        # Restore model state
        model.ngram_counts = model_data['ngram_counts']
        model.total_unigrams = model_data['total_unigrams']

        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        raise
    except IOError as e:
        print(f"Error loading model: {e}")
        raise


def tokenize(text):
    """
    Tokenizes text into words and periods.

    Args:
        text (str): Input text to tokenize

    Returns:
        list: List of lowercase tokens matching words or periods
    """
    return re.findall(r"\b[a-zA-Z0-9]+\b|[.]", text.lower())


def generate_text(model, context, num_tokens):
    """
    Generates text by repeatedly sampling from the model.

    Args:
        model (CountLanguageModel): Trained language model
        context (list): Initial context tokens
        num_tokens (int): Number of tokens to generate

    Returns:
        str: Generated text including initial context
    """
    # Start with the provided context
    generated = list(context)

    # Generate new tokens until we reach the desired length
    while len(generated) - len(context) < num_tokens:
        # Use the last n-1 tokens as context for prediction
        next_token = model.predict_next_token(generated[-(model.n-1):])
        generated.append(next_token)

        # Stop if we've generated enough tokens AND found a period
        # This helps ensure complete sentences
        if len(generated) - len(context) >= num_tokens and next_token == '.':
            break

    # Join tokens with spaces to create readable text
    return ' '.join(generated)


def compute_perplexity(model, tokens, context_size):
    """
    Computes perplexity of the model on given tokens.

    Args:
        model (CountLanguageModel): Trained language model
        tokens (list): List of tokens to evaluate on
        context_size (int): Maximum context size to consider

    Returns:
        float: Perplexity score (lower is better)
    """
    # Handle empty token list
    if not tokens:
        return float('inf')

    # Initialize log likelihood accumulator
    total_log_likelihood = 0
    num_tokens = len(tokens)

    # Calculate probability for each token given its context
    for i in range(num_tokens):
        # Get appropriate context window, handling start of sequence
        context_start = max(0, i - context_size)
        context = tuple(tokens[context_start:i])
        token = tokens[i]

        # Get probability of this token given its context
        probability = model.get_probability(token, context)

        # Add log probability to total (using log for numerical stability)
        total_log_likelihood += math.log(probability)

    # Calculate average log likelihood
    average_log_likelihood = total_log_likelihood / num_tokens

    # Convert to perplexity: exp(-average_log_likelihood)
    # Lower perplexity indicates better model performance
    perplexity = math.exp(-average_log_likelihood)
    return perplexity


class CountLanguageModel:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = [{} for _ in range(n)]
        self.total_unigrams = 0

    def predict_next_token(self, context):
        for n in range(self.n, 1, -1):
            if len(context) >= n - 1:
                context_n = tuple(context[-(n-1):])
                counts = self.ngram_counts[n-1].get(context_n)
                if counts:
                    return max(counts.items(), key=lambda x: x[1])[0]
        unigram_counts = self.ngram_counts[0].get(())
        if unigram_counts:
            return max(unigram_counts.items(), key=lambda x: x[1])[0]
        return None

    def get_probability(self, token, context):
        for n in range(self.n, 1, -1):
            if len(context) >= n - 1:
                context_n = tuple(context[-(n - 1):])
                counts = self.ngram_counts[n - 1].get(context_n)
                if counts:
                    total = sum(counts.values())
                    count = counts.get(token, 0)
                    if count > 0:
                        return count / total
        unigram_counts = self.ngram_counts[0].get(())
        count = unigram_counts.get(token, 0)
        V = len(unigram_counts)
        return (count + 1) / (self.total_unigrams + V)


def train(model: CountLanguageModel, tokens: list):
    for n in range(1, model.n + 1):
        counts = model.ngram_counts[n-1]
        for i in range(len(tokens) - n + 1):
            context = tuple(tokens[i:i+n-1])
            next_token = tokens[i+n-1]
            if context not in counts:
                counts[context] = defaultdict(int)
            counts[context][next_token] += 1

    model.total_unigrams = len(tokens)


def train_model():
    # Initialize random seeds for reproducibility
    set_seed(42)
    n = get_hyperparameters()
    model_name = "count_model"

    # Download and prepare the Brown corpus
    data_url = "https://www.thelmbook.com/data/brown"
    train_corpus, test_corpus = download_and_prepare_data(data_url)

    # Train the model and evaluate its performance
    print("\nTraining the model...")
    model = CountLanguageModel(n)
    train(model, train_corpus)
    print("\nModel training complete.")

    perplexity = compute_perplexity(model, test_corpus, n)
    print(f"\nPerplexity on test corpus: {perplexity:.2f}")

    save_model(model, model_name)


def test_model(model_name):
    model = load_model(model_name)

    # Test the model with some example contexts
    contexts = [
        "i will build a",
        "the best place to",
        "she was riding a"
    ]

    # Generate completions for each context
    for context in contexts:
        tokens = tokenize(context)
        next_token = model.predict_next_token(tokens)
        print(f"\nContext: {context}")
        print(f"Next token: {next_token}")
        print(f"Generated text: {generate_text(model, tokens, 10)}")


if __name__ == '__main__':
    train_model()
    test_model('count_model')
