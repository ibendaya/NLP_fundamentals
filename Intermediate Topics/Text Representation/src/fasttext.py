import random
import re
from collections import Counter, defaultdict

import numpy as np


class FastText:
    def __init__(
        self,
        vector_size=50,
        window_size=2,
        learning_rate=0.01,
        epochs=10,
        min_n=3,
        max_n=6,
        negative_samples=5,
    ):
        """
        Initializes the FastText model.

        Parameters:
        - vector_size (int): Dimensionality of word vectors.
        - window_size (int): Context window size.
        - learning_rate (float): Learning rate for gradient descent.
        - epochs (int): Number of training iterations.
        - min_n (int): Minimum length of subword n-grams.
        - max_n (int): Maximum length of subword n-grams.
        - negative_samples (int): Number of negative samples for each positive sample.
        """
        self.vector_size = vector_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_n = min_n
        self.max_n = max_n
        self.negative_samples = negative_samples
        self.vocab = {}
        self.word_to_index = {}
        self.index_to_word = {}
        self.subword_to_index = {}
        self.W = None  # Word vectors
        self.W_sub = None  # Subword vectors

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
        Builds the vocabulary and subword n-grams.

        Parameters:
        - corpus (list of str): A list of documents (each document is a string).
        """
        tokens = []
        for doc in corpus:
            tokens.extend(self.preprocess(doc))

        # Count word frequencies
        word_counts = Counter(tokens)

        # Build vocab and initialize word-to-index mappings
        self.word_to_index = {word: i for i, word in enumerate(word_counts)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        self.vocab = set(self.word_to_index.keys())

        # Build subword n-grams
        subwords = set()
        for word in self.vocab:
            subwords.update(self.get_subwords(word))

        self.subword_to_index = {sub: i for i, sub in enumerate(subwords)}

        # Initialize embeddings
        vocab_size = len(self.vocab)
        subword_size = len(self.subword_to_index)
        self.W = np.random.rand(vocab_size, self.vector_size)  # Word embeddings
        self.W_sub = np.random.rand(
            subword_size, self.vector_size
        )  # Subword embeddings

    def get_subwords(self, word):
        """
        Generates subword n-grams for a given word.

        Parameters:
        - word (str): The input word.

        Returns:
        - subwords (set): A set of subword n-grams.
        """
        word = f"<{word}>"
        subwords = set()
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(word) - n + 1):
                subwords.add(word[i : i + n])
        return subwords

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
                start = max(0, idx - self.window_size)
                end = min(len(tokens), idx + self.window_size + 1)
                for context_word in tokens[start:idx] + tokens[idx + 1 : end]:
                    pairs.append((word, context_word))
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
        Trains the FastText model using the Skip-gram architecture with subword embeddings.

        Parameters:
        - corpus (list of str): A list of documents.
        """
        training_data = self.generate_training_data(corpus)
        subword_indices = {
            word: [self.subword_to_index[sub] for sub in self.get_subwords(word)]
            for word in self.vocab
        }

        for epoch in range(self.epochs):
            loss = 0
            for target_word, context_word in training_data:
                target_idx = self.word_to_index[target_word]
                context_idx = self.word_to_index[context_word]

                # Get subword indices for the target word
                target_subword_indices = subword_indices[target_word]

                # Compute target word embedding as the sum of its subword embeddings
                target_embedding = np.sum(self.W_sub[target_subword_indices], axis=0)

                # Positive sample
                context_embedding = self.W[context_idx]
                positive_score = self.sigmoid(
                    np.dot(target_embedding, context_embedding)
                )
                loss += -np.log(positive_score)

                # Gradient update for positive sample
                grad = self.learning_rate * (1 - positive_score)
                self.W[context_idx] += grad * target_embedding
                for sub_idx in target_subword_indices:
                    self.W_sub[sub_idx] += grad * context_embedding

                # Negative sampling
                for _ in range(self.negative_samples):
                    negative_idx = random.randint(0, len(self.vocab) - 1)
                    negative_embedding = self.W[negative_idx]
                    negative_score = self.sigmoid(
                        -np.dot(target_embedding, negative_embedding)
                    )
                    loss += -np.log(negative_score)

                    # Gradient update for negative sample
                    grad_neg = self.learning_rate * (1 - negative_score)
                    for sub_idx in target_subword_indices:
                        self.W_sub[sub_idx] -= grad_neg * negative_embedding
                    self.W[negative_idx] -= grad_neg * target_embedding

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

    def get_embedding(self, word):
        """
        Retrieves the embedding for a given word, including OOV words.

        Parameters:
        - word (str): The target word.

        Returns:
        - embedding (numpy array): The word embedding vector.
        """
        if word in self.word_to_index:
            return self.W[self.word_to_index[word]]
        else:
            subword_indices = [
                self.subword_to_index[sub]
                for sub in self.get_subwords(word)
                if sub in self.subword_to_index
            ]
            if subword_indices:
                return np.sum(self.W_sub[subword_indices], axis=0)
            else:
                raise ValueError(f"Word '{word}' not in vocabulary or subword list.")
