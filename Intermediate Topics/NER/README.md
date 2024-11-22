**Named Entity Recognition (NER)** is the task of identifying and classifying named entities in text into predefined categories such as names of people, organizations, locations, dates, etc. This is useful for:

- information extraction
- building knowledge graphs
- question answering systems

Preprocessing recommendations typically include:
- **Text cleaning**: lowercasing is optional depending on case sensitivity, removal of non-alphabetical characters is optional, depending on if entities are defined by a specific formatting
- **Tokenization**: Break the text into smaller units (tokens), typically words or sub-words, using libraries like NLTK or SpaCy.
- **Label encoding**: for supervised learning, map each token to its entity label
- **Padding and truncation**: ensure uniform sequence lengths for batch training

Key metrics usually include Accuracy, precision, recall, and F1-Score. However, Entity-level F1 score can be used to evaluate the performance of the model by considering entities as a whole (exact span matches)