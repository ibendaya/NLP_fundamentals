# Deep Learning Basics

## Neural Networks
- **Definition**: Neural networks are computational models inspired by the human brain, consisting of layers of interconnected nodes (neurons).
- **Components**:
  - Input Layer: Takes input features.
  - Hidden Layers: Process inputs with weights and activation functions.
  - Output Layer: Produces predictions.

## Convolutional Neural Networks (CNNs)
- **Purpose**: Primarily used for image data but can be applied to text for specific tasks (e.g., text classification).
- **Key Components**:
  - Convolution Layer: Extracts features using filters.
  - Pooling Layer: Reduces spatial dimensions.
  - Fully Connected Layer: Maps extracted features to outputs.

## Recurrent Neural Networks (RNNs)
- **Purpose**: Designed for sequential data (e.g., text, time series).
- **Key Feature**: Maintains a hidden state that captures information about previous time steps.
- **Challenges**: Vanishing gradient problem.

## Long Short-Term Memory (LSTM) Networks
- **Improvement Over RNNs**: Introduces gates (forget, input, output) to manage long-term dependencies.
- **Applications**: Sentiment analysis, text generation, machine translation.

## Gated Recurrent Units (GRUs)
- **Simpler Alternative to LSTMs**: Combines forget and input gates into a single update gate.
- **Advantage**: Requires fewer parameters.

## Attention Mechanisms
- **Purpose**: Focuses on the most relevant parts of the input sequence when producing output.
- **Applications**: Machine translation, text summarization.

## Seq2Seq Models
- **Definition**: Encoder-decoder architectures for sequence-to-sequence tasks.
- **Components**:
  - Encoder: Encodes input sequence into a fixed-length context vector.
  - Decoder: Generates the output sequence from the context vector.
