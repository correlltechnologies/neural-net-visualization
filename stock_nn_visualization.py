import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATASET_NAME = "pmoe7/SP_500_Stocks_Data-ratios_news_price_10_yrs"
# Using a small subset for visualization purposes
NUM_SAMPLES = 200 
HIDDEN_SIZE_1 = 8
HIDDEN_SIZE_2 = 6
OUTPUT_SIZE = 3  # Up, Down, Neutral

# --- 1. Data Loading & Preprocessing ---
def load_and_process_data():
    print(f"Loading dataset: {DATASET_NAME}...")
    try:
        # Load dataset (streaming or full load depending on size, this one is likely manageable)
        dataset = load_dataset(DATASET_NAME, split="train")
        
        # Convert to Pandas DataFrame for easier manipulation
        df = dataset.to_pandas()
        
        # Filter for a specific stock to make it a coherent time series (e.g., AAPL)
        # If 'Symbol' or similar column exists. If not, we'll just take the first N rows.
        if 'Symbol' in df.columns:
            print("Filtering for AAPL...")
            df = df[df['Symbol'] == 'AAPL']
        
        # Select features
        # Assuming standard column names; adjust if necessary based on actual dataset schema
        # Common names: 'Open', 'High', 'Low', 'Close', 'Volume'
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # verify columns exist, strict checking
        available_cols = [c for c in feature_cols if c in df.columns]
        if not available_cols:
             # Fallback if specific columns aren't found (using first 5 numeric columns)
             print("Specific columns not found, using first 5 numeric columns.")
             numeric_df = df.select_dtypes(include=[np.number])
             available_cols = numeric_df.columns[:5].tolist()
        
        print(f"Using features: {available_cols}")
        data = df[available_cols].values
        
        # Create target (Price movement: 0=Down, 1=Neutral, 2=Up)
        # Simple heuristic: Close price comparison
        # We need 'Close' for this. If not available, we make a dummy target.
        if 'Close' in df.columns:
            closes = df['Close'].values
            targets = []
            for i in range(len(closes) - 1):
                change = (closes[i+1] - closes[i]) / closes[i]
                if change > 0.005: targets.append(2) # Up
                elif change < -0.005: targets.append(0) # Down
                else: targets.append(1) # Neutral
            targets.append(1) # Last one
            targets = np.array(targets)
        else:
            targets = np.random.randint(0, 3, size=len(data))

        # Normalize
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data)
        
        return data_normalized[:NUM_SAMPLES], targets[:NUM_SAMPLES]

    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback to dummy data if dataset loading fails (e.g., no internet or dataset changes)
        print("Generating dummy data...")
        return np.random.randn(NUM_SAMPLES, 5), np.random.randint(0, 3, NUM_SAMPLES)

# --- 2. Neural Network Model ---
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, output_size)
        self.activations = {}

    def forward(self, x):
        x = self.fc1(x)
        self.activations['fc1'] = torch.relu(x) # Capture activation
        x = F.relu(x)
        
        x = self.fc2(x)
        self.activations['fc2'] = torch.tanh(x) # Capture activation
        x = torch.tanh(x)
        
        x = self.out(x)
        self.activations['out'] = torch.softmax(x, dim=1) # Capture activation
        return x

# --- 3. Visualization ---
def visualize_network(X, y, model):
    input_size = X.shape[1]
    
    # Create Graph
    G = nx.DiGraph()
    
    # Define layers and nodes
    layers = [input_size, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE]
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
    
    pos = {}
    node_colors = []
    
    # Calculate positions
    # Use a fixed layout: x=layer_index, y=centered_node_index
    for i, layer_size in enumerate(layers):
        x = i * 2  # Spacing between layers
        y_offset = (layer_size - 1) / 2.0
        for j in range(layer_size):
            node_id = f"{i}_{j}"
            G.add_node(node_id, layer=i)
            pos[node_id] = (x, j - y_offset)
    
    # Add edges
    for i in range(len(layers) - 1):
        for j in range(layers[i]):
            for k in range(layers[i+1]):
                u = f"{i}_{j}"
                v = f"{i+1}_{k}"
                G.add_edge(u, v)

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title("Neural Network Activation Flow")
    
    # Draw static edges (connections)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.3)
    
    # Draw nodes (initial state)
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color='blue', node_size=300, alpha=0.8)
    
    # Labels for layers
    for i, name in enumerate(layer_names):
        plt.text(i * 2, -max(layers)/2 - 1, name, ha='center', fontsize=12, fontweight='bold')

    # Animation function
    def update(frame):
        input_row = torch.FloatTensor(X[frame]).unsqueeze(0)
        
        # Forward pass
        _ = model(input_row)
        
        # Collect activations
        # Input layer
        act_vals = list(input_row.detach().numpy().flatten())
        
        # Hidden & Output layers
        act_vals.extend(model.activations['fc1'].detach().numpy().flatten())
        act_vals.extend(model.activations['fc2'].detach().numpy().flatten())
        act_vals.extend(model.activations['out'].detach().numpy().flatten())
        
        # Normalize activations for color mapping (0 to 1)
        # Using sigmoid or simple min-max scaling for visualization
        colors = []
        sizes = []
        for val in act_vals:
            # Simple visualization mapping
            # Light up if positive, dim if negative/zero
            intensity = 1 / (1 + np.exp(-val)) # Sigmoid to keep in 0-1
            colors.append((intensity, 0, 1-intensity)) # Red to Blue gradient
            sizes.append(200 + intensity * 300)
            
        nodes.set_color(colors)
        nodes.set_sizes(sizes)
        
        ax.set_title(f"Time Step: {frame} | Prediction: {['Down', 'Neutral', 'Up'][np.argmax(act_vals[-3:])]}")
        return nodes,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(X), interval=100, blit=False)
    
    print("Saving animation to 'results/nn_activation.gif'...")
    try:
        ani.save('results/nn_activation.gif', writer='pillow', fps=10)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"Could not save animation: {e}")
        print("Displaying plot instead (if environment supports it)...")
        plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    X, y = load_and_process_data()
    print(f"Data shape: {X.shape}")
    
    # Initialize Model
    model = StockPredictor(X.shape[1], HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE)
    
    # Visualize
    visualize_network(X, y, model)
