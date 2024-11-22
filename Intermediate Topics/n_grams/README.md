# Primer on N-grams and Markov Language Models

## **1. N-grams**

**N-grams** are a contiguous sequence of \(n\) items (words, characters, etc.) from a given text or speech sequence. They are a foundational concept in natural language processing (NLP) for modeling sequences and predicting the next item in a sequence.

- **Unigram**: \(n = 1\) (single word or token)
  - Example: `["The", "cat", "sat"]`
- **Bigram**: \(n = 2\) (sequence of two words)
  - Example: `["The cat", "cat sat"]`
- **Trigram**: \(n = 3\) (sequence of three words)
  - Example: `["The cat sat"]`

### **Applications**:
- Text prediction (e.g., autocomplete, next-word prediction)
- Machine translation
- Speech recognition
- Sentiment analysis

### **Probability of N-grams**:
The probability of a sequence of words in an N-gram model is computed as:
\[
P(w_1, w_2, \ldots, w_m) \approx \prod_{i=1}^{m} P(w_i | w_{i-n+1}, \ldots, w_{i-1})
\]
For a bigram model:
\[
P(w_1, w_2, \ldots, w_m) \approx P(w_1) P(w_2 | w_1) P(w_3 | w_2) \ldots
\]

---

## **2. Markov Language Models**

Markov language models build on N-grams by leveraging the **Markov assumption**: the probability of the next word depends only on a fixed number of previous words (usually \(n-1\), where \(n\) is the N-gram size).

### **Key Concepts**:
- **Markov Assumption**: Future state (word) depends only on the current or recent past states (previous words).
- **First-Order Markov Model**: Probability of a word depends only on the previous word (bigram model).
  \[
  P(w_i | w_1, w_2, \ldots, w_{i-1}) \approx P(w_i | w_{i-1})
  \]
- **Second-Order Markov Model**: Probability of a word depends on the two previous words (trigram model).
  \[
  P(w_i | w_1, w_2, \ldots, w_{i-1}) \approx P(w_i | w_{i-2}, w_{i-1})
  \]

---

## **3. Training Markov Language Models**

1. **Collect Data**: Use a large text corpus.
2. **Count Frequencies**:
   - For bigram: Count how often pairs of words occur.
3. **Estimate Probabilities**:
   - \(P(w_i | w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\text{Count}(w_{i-1})}\)

---

## **4. Smoothing Techniques**

Real-world data can result in zero probabilities for unseen N-grams. Smoothing addresses this:
- **Laplace Smoothing**: Adds a small constant to each count.
  \[
  P(w_i | w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i) + 1}{\text{Count}(w_{i-1}) + V}
  \]
  Where \(V\) is the vocabulary size.
- **Good-Turing Smoothing**: Adjusts probabilities based on the frequency of rare events.

---

## **5. Limitations of N-gram Models**

- **Data Sparsity**: N-gram models require large amounts of data, especially for higher-order N-grams.
- **Fixed Context Window**: They can only consider a limited number of previous words.
- **Lack of Generalization**: Cannot capture long-range dependencies.

---

## **6. Modern Context**

N-grams and Markov models have largely been replaced by neural models like **RNNs**, **LSTMs**, and **Transformers** in most NLP applications. However, they remain valuable for:
- Computational efficiency in small-scale tasks
- Understanding the basics of sequence modeling
