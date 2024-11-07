import re
from collections import Counter


class BagOfWords:
    def __init__(self, stopwords=None):
        """
        Initializes the Bag of Words model.

        Parameters:
        - stopwords (set): A set of words to ignore. Default is None.
        """
        self.stopwords = stopwords if stopwords else set()
        self.vocab = set()
        self.word_to_index = {}

    def preprocess(self, text):
        """
        Preprocesses the input text (tokenization, lowercasing, removing
        stopwords).

        Parameters:
        - text (str): The raw text string.

        Returns:
        - tokens (list): A list of preprocessed tokens.
        """
        # Lowercase the text and remove special characters
        text = text.lower()
        text = re.sub(r"\W+", " ", text)
        tokens = text.split()

        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stopwords]
        return tokens

    def build_vocab(self, corpus):
        """
        Builds the vocabulary from a corpus of documents.

        Parameters:
        - corpus (list of str): A list of documents (each document is a
        string).
        """
        for document in corpus:
            tokens = self.preprocess(document)
            self.vocab.update(tokens)

        # Assign an index to each unique word
        self.word_to_index = {w: i for i, w in enumerate(sorted(self.vocab))}

    def vectorize(self, document):
        """
        Converts a document into its Bag of Words vector representation.

        Parameters:
        - document (str): A single document (string).

        Returns:
        - vector (list): A list representing the BoW vector for the document.
        """
        tokens = self.preprocess(document)
        vector = [0] * len(self.vocab)

        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            if token in self.word_to_index:
                index = self.word_to_index[token]
                vector[index] = count

        return vector

    def transform(self, corpus):
        """
        Transforms a corpus into BoW vectors.

        Parameters:
        - corpus (list of str): A list of documents.

        Returns:
        - vectors (list of list): A list of BoW vectors.
        """
        return [self.vectorize(doc) for doc in corpus]
