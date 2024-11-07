import math
import re
from collections import Counter, defaultdict


class TFIDF:
    def __init__(self, stopwords=None):
        """
        Initializes the TF-IDF model.

        Parameters:
        - stopwords (set): A set of words to ignore. Default is None.
        """
        self.stopwords = stopwords if stopwords else set()
        self.vocab = set()
        self.doc_freq = defaultdict(int)  # Stores document frequency for each term
        self.idf = {}  # Stores inverse document frequency for each term

    def preprocess(self, text):
        """
        Preprocesses the input text (tokenization, lowercasing, removing stopwords).

        Parameters:
        - text (str): The raw text string.

        Returns:
        - tokens (list): A list of preprocessed tokens.
        """
        text = text.lower()
        text = re.sub(r"\W+", " ", text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stopwords]
        return tokens

    def build_vocab_and_idf(self, corpus):
        """
        Builds the vocabulary and computes the IDF values from a corpus.

        Parameters:
        - corpus (list of str): A list of documents (each document is a string).
        """
        num_docs = len(corpus)
        for document in corpus:
            tokens = set(
                self.preprocess(document)
            )  # Use set to count unique tokens in a document
            for token in tokens:
                self.vocab.add(token)
                self.doc_freq[token] += 1

        # Compute IDF for each term
        self.idf = {
            term: math.log((1 + num_docs) / (1 + self.doc_freq[term])) + 1
            for term in self.vocab
        }

    def compute_tf(self, document):
        """
        Computes the term frequency (TF) for a single document.

        Parameters:
        - document (str): A single document (string).

        Returns:
        - tf (dict): A dictionary of term frequencies.
        """
        tokens = self.preprocess(document)
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        tf = {term: count / total_tokens for term, count in token_counts.items()}
        return tf

    def vectorize(self, document):
        """
        Converts a document into its TF-IDF vector representation.

        Parameters:
        - document (str): A single document (string).

        Returns:
        - vector (dict): A dictionary representing the TF-IDF vector for the
        document.
        """
        tf = self.compute_tf(document)
        tfidf_vector = {
            term: tf.get(term, 0) * self.idf.get(term, 0) for term in self.vocab
        }
        return tfidf_vector

    def transform(self, corpus):
        """
        Transforms a corpus into TF-IDF vectors.

        Parameters:
        - corpus (list of str): A list of documents.

        Returns:
        - vectors (list of dict): A list of TF-IDF vectors for each document.
        """
        return [self.vectorize(doc) for doc in corpus]

    def vector_to_dense(self, vector):
        """
        Converts a sparse TF-IDF vector to a dense list of scores.

        Parameters:
        - vector (dict): A sparse dictionary representing the TF-IDF vector.

        Returns:
        - dense_vector (list): A dense list of TF-IDF scores.
        """
        return [vector[term] for term in sorted(self.vocab)]
