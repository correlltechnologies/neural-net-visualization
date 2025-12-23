# Neural Network Model Explanation

## Overview
This project visualizes a simple Feed-Forward Neural Network (Multilayer Perceptron) processing financial time-series data.

## Architecture

### 1. Input Layer
- **Nodes**: 5 (representing 5 features)
- **Features**:
  1. **Open**: The stock price at the market open.
  2. **High**: The highest price of the day.
  3. **Low**: The lowest price of the day.
  4. **Close**: The stock price at market close.
  5. **Volume**: The number of shares traded.
- **Normalization**: All inputs are normalized (scaled) using `StandardScaler` to have a mean of 0 and variance of 1. This ensures the neural network can process the data effectively.

### 2. Hidden Layers
- **Hidden Layer 1**: 8 Neurons with **ReLU** (Rectified Linear Unit) activation.
- **Hidden Layer 2**: 6 Neurons with **Tanh** (Hyperbolic Tangent) activation.
- **Function**: These layers extract patterns and non-linear relationships from the input features.

### 3. Output Layer
- **Nodes**: 3
- **Classes**:
  1. **Down**: Prediction that the stock price will decrease.
  2. **Neutral**: Prediction that the price will remain relatively stable.
  3. **Up**: Prediction that the stock price will increase.
- **Activation**: **Softmax**. This converts the raw output scores into probabilities (summing to 1).

## Visualization Details
- **Nodes**: Circles represent neurons. Their color intensity and size change based on their **activation value** (how "excited" the neuron is by the input).
- **Particles**: Moving dots represent the flow of information (signals) from one layer to the next.
- **Connections**: Lines represent the weights connecting neurons.
- **Live Output**: The text above the nodes shows the exact activation value at that moment.

## "Why does it make the same decisions?"
If you observe that the network makes repetitive predictions (e.g., always "Neutral" or "Up"), this is likely because:
1. **Random Initialization**: The network is initialized with random weights and has not been *trained* for many epochs on this specific data. It is essentially "guessing" based on its initial random structure.
2. **Class Imbalance**: Financial data often has a lot of "noise" or small movements (Neutral), leading an untrained or simple model to bias towards the most common state.
3. **Visualization Focus**: The primary goal of this tool is to visualize *how* data flows and activates a network, rather than to provide an accurate trading algorithm.

To improve predictions, one would need to implement a full training loop (`optimizer.step()`, loss calculation) and train the model for several epochs before visualizing.
