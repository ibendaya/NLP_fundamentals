## Folder structure
```
Text Representation/
│
├── data/
│   └── (Scripts to download/process data)
├── notebooks/
│   └── (Jupyter notebooks with implementations and results)
├── src/
│   ├── bow.py
│   ├── tfidf.py
│   ├── word2vec.py
│   ├── glove.py
│   └── fasttext.py
├── notes.md
└── sandbox.ipynb
```

## Overview of Text Representation in NLP
In Natural Language Processing (NLP), converting text into numerical representations is essential for machine learning models to process and analyze language. Some of the fundamental text representation methods include:

### 1. Bag of Words (BoW)
Concept: Treats each document as a collection of words without considering the order.

Implementation:
 - Create a vocabulary of unique words from the entire corpus.
 - Represent each document as a vector where each dimension corresponds to a word's frequency in the document.
Pros:
 - Simple and effective for small datasets.
Cons:
 - Ignores word order and semantics.
 - Results in sparse, high-dimensional vectors for large vocabularies.

### 2. Term Frequency-Inverse Document Frequency (TF-IDF)
Concept: Enhances BoW by assigning weights to words based on their importance in a document relative to the entire corpus.
- TF: Frequency of a word in a document.
- IDF: Logarithm of the total number of documents divided by the number of documents containing the word.
- TF-IDF Score: 
```
TF-IDF(𝑤,𝑑)=TF(𝑤,𝑑)×IDF(𝑤)
```
Pros:
- Reduces the impact of commonly occurring words (like "the", "is") that provide little unique information.
Cons:
- Still results in sparse vectors.
- Ignores semantic relationships between words.

### 3. Word Embeddings
These methods capture semantic meaning and relationships by representing words in dense, low-dimensional vectors.

#### a. Word2Vec
Concept: Predicts a word given its context (CBOW) or predicts context given a word (Skip-gram).

Training Objective:
- CBOW: Maximize probability of a word given its surrounding words.
- Skip-gram: Maximize probability of surrounding words given a word.

Pros:
- Captures semantic relationships (e.g., "king" - "man" + "woman" ≈ "queen").
- Efficient for large corpora.
Cons:
- Word embeddings are static (each word has a single vector regardless of context).

#### b. GloVe (Global Vectors for Word Representation)
Concept: Focuses on the co-occurrence matrix of words across the entire corpus.

Training Objective: Factorizes the word co-occurrence matrix into word vectors.

Pros:
- Captures both local context (like Word2Vec) and global statistics.
Cons:
- Similar to Word2Vec, embeddings are context-independent.

#### c. FastText
Concept: Builds on Word2Vec by breaking words into character-level n-grams.

Key Feature: Embeddings are learned for n-grams and combined for out-of-vocabulary (OOV) words.

Pros:
- Handles OOV words better by leveraging subword information.
- Useful for morphologically rich languages.
Cons:
- Increased training complexity compared to Word2Vec.

### 4. Contextual Embeddings
These methods generate dynamic embeddings based on the context of the word in the sentence.

#### a. ELMo (Embeddings from Language Models)
Concept: Learns contextualized embeddings by training a deep bidirectional language model.
Pros:
- Captures context-sensitive word representations.
Cons:
- Computationally expensive.

#### b. BERT (Bidirectional Encoder Representations from Transformers)
Concept: Uses the Transformer architecture to provide context-aware embeddings for words in sentences.

Training Objective:
- Masked Language Modeling (MLM): Predicts masked words in a sentence.
- Next Sentence Prediction (NSP): Determines if one sentence follows another.
Pros:
- Dynamic embeddings, capturing nuanced meanings depending on context.
Cons:
- Requires significant computational resources.

#### c. GPT (Generative Pretrained Transformer)
Concept: Focuses on left-to-right context during pretraining, optimizing for generating text.

Pros:
- Effective for generative tasks.
Cons:
- Contextual understanding might be less bidirectional compared to BERT.