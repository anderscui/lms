# coding=utf-8
import gzip
import json
import random

import requests

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def set_seed(seed):
    """
    Sets random seed for reproducibility.

    Args:
        seed (int): Seed value for random number generation
    """
    random.seed(seed)

def download_and_split_data(data_url, test_ratio=0.1):
    """
    Downloads emotion classification dataset from URL and splits into train/test sets.
    Handles decompression and JSON parsing of the raw data.

    Args:
        data_url (str): URL of the gzipped JSON dataset
        test_ratio (float): Proportion of data to use for testing (default: 0.1)

    Returns:
        tuple: (X_train, y_train, X_test, y_test) containing:
            - X_train, X_test: Lists of text examples for training and testing
            - y_train, y_test: Lists of corresponding emotion labels
    """
    # Download and decompress the dataset
    response = requests.get(data_url)
    content = gzip.decompress(response.content).decode()

    # Parse JSON lines into list of dictionaries
    dataset = [json.loads(line) for line in content.splitlines()]

    # Shuffle dataset for random split
    random.shuffle(dataset)

    # Split into train and test sets
    split_index = int(len(dataset) * (1 - test_ratio))
    train, test = dataset[:split_index], dataset[split_index:]

    # Separate text and labels
    X_train = [item["text"] for item in train]
    y_train = [item["label"] for item in train]
    X_test = [item["text"] for item in test]
    y_test = [item["label"] for item in test]

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    set_seed(42)

    data_url = "https://www.thelmbook.com/data/emotions"
    X_train_text, y_train, X_test_text, y_test = download_and_split_data(data_url, test_ratio=0.1)
    print(f'{len(X_train_text)} train items, {len(X_test_text)} test items loaded')

    # vectorizer = CountVectorizer(max_features=10_000, binary=True)
    vectorizer = CountVectorizer(max_features=20_000, binary=True, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print(f'train data: {X_train.shape}, test data: {X_test.shape}')

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f'training accuracy: {train_accuracy * 100:.2f}')
    print(f'test accuracy: {test_accuracy * 100:.2f}')
