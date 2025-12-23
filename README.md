# Neural Network Visualization for Stock Data

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange)
![Status](https://img.shields.io/badge/Status-Prototype-green)

A real-time visualization of a Neural Network processing stock market data. This tool visualizes how data flows through a simple Multi-Layer Perceptron (MLP), showing neuron activations and signal propagation.

![Animation Demo](results/nn_activation.gif)

## Features

- **Live Data Flow**: Watch "particles" travel between layers representing data transmission.
- **Real-time Activations**: Neurons light up and change size based on their activation values (ReLU/Tanh/Softmax).
- **Interactive**: Opens a window to watch the visualization live.
- **Customizable**: Choose any stock ticker from the S&P 500 dataset.
- **Detailed Metrics**: See the exact activation values for every neuron.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/correlltechnologies/neural-net-visualization.git
   cd neural-net-visualization
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the visualization script. By default, it uses AAPL data.

```bash
python stock_nn_visualization.py
```

To visualize a different stock:

```bash
python stock_nn_visualization.py --ticker MSFT
```

## How it Works

See [model_explanation.md](model_explanation.md) for a detailed breakdown of the neural network architecture, input features, and output classes.
