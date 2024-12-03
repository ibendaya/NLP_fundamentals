# GPT Language Model Implementation

The jupyter notebook was written following (Karpathy's video lectures)[https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1s]

This repository contains the implementation of a character-level GPT (Generative Pre-trained Transformer) language model built using PyTorch. The model is trained on a small dataset, specifically the Tiny Shakespeare text, to generate human-like text based on previous input characters. 

## Key Features

- **Character-level language modeling**: The model learns to predict the next character in a sequence, using a transformer-based architecture.
- **Customizable architecture**: The model supports configurable hyperparameters such as the number of attention heads, layers, embedding dimensions, and more.
- **Text generation**: After training, the model can generate new text based on a given input prompt by sampling from the predicted distribution over characters.
- **Evaluation**: The model's performance can be evaluated on both training and validation sets during the training process.

## Training the Model

The model is trained on the Tiny Shakespeare dataset (`tiny_shakespear.txt`). The training splits the data into a training set (90%) and a validation set (10%). Training proceeds for a set number of iterations (`max_iters`) with evaluation intervals (`eval_interval`) to monitor loss on both the training and validation sets.

### Hyperparameters
- `batch_size`: Number of independent sequences processed in parallel (default: 16)
- `block_size`: Maximum context length for predictions (default: 32)
- `max_iters`: Number of iterations for training (default: 5000)
- `learning_rate`: Learning rate for the optimizer (default: 1e-3)
- `eval_interval`: Interval for evaluation (default: 100)
- `n_embd`: Embedding dimension size (default: 64)
- `n_head`: Number of attention heads (default: 4)
- `n_layer`: Number of transformer layers (default: 8)
- `dropout`: Dropout rate (default: 0.0)

### Dataset
The model is trained on the Tiny Shakespeare dataset. The data is split into training and validation sets. The training data makes up 90% of the total, while the remaining 10% is used for validation.

### Model Overview
The architecture of the model is based on the transformer architecture, consisting of:
1. **Token Embedding Layer**: Converts input tokens into dense vectors of fixed size.
2. **Position Embedding Layer**: Adds positional information to the token embeddings to account for the order of tokens in the sequence.
3. **Transformer Blocks**: A sequence of layers consisting of multi-head self-attention and feed-forward neural networks.
4. **Final Layer**: A linear layer to predict the next token in the sequence.

### Training Loop
The training loop involves:
1. Sampling batches from the training set.
2. Feeding the batches into the model.
3. Calculating the loss and backpropagating the gradients.
4. Optimizing the model parameters using AdamW.

Every `eval_interval` steps, the model's performance is evaluated on both the training and validation sets.

### Text Generation
After training, the model can generate new text based on a given prompt. This is done by feeding the model a sequence of tokens and using the model's output to predict the next token in the sequence. The process is repeated for a predefined number of steps, generating new tokens each time.

## Conclusion

This implementation demonstrates how a basic transformer architecture can be used for text generation. By training on the Tiny Shakespeare dataset, the model learns to generate coherent and contextually relevant sequences of text. The flexibility of the model's hyperparameters allows for experimentation with different configurations to improve performance.