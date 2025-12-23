Goal: Build a Python neural network visualization that shows neurons being created and activating ("lighting up") as a time-series input flows through.

Dataset:
- Use the Hugging Face dataset "pmoe7/SP_500_Stocks_Data-ratios_news_price_10_yrs"
  (daily stock price + fundamentals + news sentiment for S&P 500). :contentReference[oaicite:2]{index=2}

Requirements:
1. Load the dataset using the Hugging Face `datasets` library.
2. Extract a subset of numerical features like: open, high, low, close, volume, and a few technical indicators if available.
3. Preprocess and normalize the data for neural network input.
4. Build a small neural network in PyTorch or TensorFlow with:
   - Input layer matching the number of features
   - One or two hidden layers (ReLU or tanh activations)
   - Final output layer with 3–5 neurons representing simple classes (e.g., predicted trend up/down/neutral).
5. Set up hooks or callbacks to capture activations for each input row.
6. Visualize the network:
   - Draw nodes and layers using a visualization library (e.g., NetworkX + Matplotlib).
   - For each time step (row) in the dataset, run a forward pass and update node visuals:
       • Node size, color intensity, or glow corresponds to activation magnitude.
       • Animate across the series (Matplotlib animation or interactive PyGame).
7. Optional: Save animation to video or allow interactive stepping.
8. Ensure the visualization clearly shows:
   - Network creation
   - Activation propagation from input → hidden → output layers
   - Output decision strength
   - Smooth time progression over stock dataset

Deliverables:
- Python script (`stock_nn_visualization.py`)
- Instructions on how to install dependencies
- Example animation output
- Comments explaining activation capture and visualization logic

Use best practices for clean, modular code.
