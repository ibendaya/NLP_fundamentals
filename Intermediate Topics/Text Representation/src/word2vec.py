import random
import re
from collections import defaultdict
from itertools import chain

import numpy as np


class Word2Vec:
    def __init__(
        self,
        vector_size=50,
        window_size=2,
        learning_rate=0.01,
        epochs=10,
        negative_samples=5,
    ):
        """
        Initializes the Word2Vec model.

        Parameters:
        - vector_size (int): Dimensionality of word vectors.
        - window_size (int): Context window size.
        - learning_rate (float): Learning rate for gradient descent.
        - epochs (int): Number of training iterations.
        - negative_samples (int): Number of negative samples for each positive sample.
        """
        self.vector_size = vector_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.vocab = {}
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_counts = defaultdict(int)
        self.W = None  # Target word vectors
        self.W_prime = None  # Context word vectors

    def preprocess(self, text):
        """
        Preprocesses the input text (tokenization, lowercasing).

        Parameters:
        - text (str): The raw text string.

        Returns:
        - tokens (list): A list of preprocessed tokens.
        """
        text = text.lower()
        text = re.sub(r"\W+", " ", text)
        tokens = text.split()
        return tokens

    def build_vocab(self, corpus):
        """
        Builds the vocabulary from the corpus and initializes word vectors.

        Parameters:
        - corpus (list of str): A list of documents (each document is a string).
        """
        tokens = list(chain.from_iterable([self.preprocess(doc) for doc in corpus]))

        # Count word frequencies
        for token in tokens:
            self.word_counts[token] += 1

        # Build vocab with unique words and their indices
        self.vocab = {word for word in self.word_counts}
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_word = {idx: w for w, idx in self.word_to_index.items()}

        # Initialize word vectors
        vocab_size = len(self.vocab)
        self.W = np.random.rand(vocab_size, self.vector_size)  # Target word vectors
        self.W_prime = np.random.rand(
            vocab_size, self.vector_size
        )  # Context word vectors

    def generate_training_data(self, corpus):
        """
        Generates training data (target-context pairs) for the Skip-gram model.

        Parameters:
        - corpus (list of str): A list of documents.

        Returns:
        - training_data (list of tuples): A list of (target, context) word pairs.
        """
        pairs = []
        for doc in corpus:
            tokens = self.preprocess(doc)
            for idx, word in enumerate(tokens):
                target_idx = self.word_to_index[word]
                start = max(0, idx - self.window_size)
                end = min(len(tokens), idx + self.window_size + 1)
                for context_word in tokens[start:idx] + tokens[idx + 1 : end]:
                    pairs.append((target_idx, self.word_to_index[context_word]))
        return pairs

    def sigmoid(self, x):
        """
        Computes the sigmoid of x.

        Parameters:
        - x (float): The input value.

        Returns:
        - sigmoid (float): Sigmoid output.
        """
        return 1 / (1 + np.exp(-x))

    def train(self, corpus):
        """
        Trains the Word2Vec model using the Skip-gram architecture and negative sampling.

        Parameters:
        - corpus (list of str): A list of documents.
        """
        training_data = self.generate_training_data(corpus)
        vocab_size = len(self.vocab)

        for epoch in range(self.epochs):
            loss = 0
            for target_idx, context_idx in training_data:
                # Positive sample
                target_vector = self.W[target_idx]
                context_vector = self.W_prime[context_idx]

                positive_score = self.sigmoid(np.dot(target_vector, context_vector))
                loss += -np.log(positive_score)

                # Gradient update for positive sample
                grad = self.learning_rate * (1 - positive_score)
                self.W[target_idx] += grad * context_vector
                self.W_prime[context_idx] += grad * target_vector

                # Negative sampling
                for _ in range(self.negative_samples):
                    negative_idx = random.randint(0, vocab_size - 1)
                    if negative_idx == context_idx:
                        continue
                    negative_vector = self.W_prime[negative_idx]

                    negative_score = self.sigmoid(
                        -np.dot(target_vector, negative_vector)
                    )
                    loss += -np.log(negative_score)

                    # Gradient update for negative sample
                    grad_neg = self.learning_rate * (1 - negative_score)
                    self.W_prime[negative_idx] -= grad_neg * target_vector
                    self.W[target_idx] -= grad_neg * negative_vector

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

    def get_embedding(self, word):
        """
        Retrieves the embedding for a given word.

        Parameters:
        - word (str): The target word.

        Returns:
        - embedding (numpy array): The word embedding vector.
        """
        idx = self.word_to_index.get(word)
        if idx is not None:
            return self.W[idx]
        else:
            raise ValueError(f"Word '{word}' not in vocabulary.")
