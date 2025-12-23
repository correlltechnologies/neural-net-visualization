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
import argparse

# --- Configuration ---
DATASET_NAME = "pmoe7/SP_500_Stocks_Data-ratios_news_price_10_yrs"
NUM_SAMPLES = 200 
HIDDEN_SIZE_1 = 8
HIDDEN_SIZE_2 = 6
OUTPUT_SIZE = 3  # Up, Down, Neutral

# --- 1. Data Loading & Preprocessing ---
def load_and_process_data(ticker="AAPL"):
    print(f"Loading dataset: {DATASET_NAME}...")
    try:
        # Specifically target the price/ratios CSV
        dataset = load_dataset(DATASET_NAME, data_files="sp500_daily_ratios_20yrs.zip", split="train")
        
        # Convert to Pandas DataFrame
        df = dataset.to_pandas()
        
        # Filter for the selected stock
        print(f"Filtering for {ticker}...")
        if 'Ticker' in df.columns:
            df = df[df['Ticker'] == ticker]
        elif 'Symbol' in df.columns:
            df = df[df['Symbol'] == ticker]
            
        if df.empty:
            print(f"Warning: No data found for {ticker}. Falling back to first available ticker.")
            if 'Ticker' in df.columns:
                first_ticker = df['Ticker'].iloc[0]
                df = df[df['Ticker'] == first_ticker]
            elif 'Symbol' in df.columns:
                 first_ticker = df['Symbol'].iloc[0]
                 df = df[df['Symbol'] == first_ticker]

        # Sort by date if available
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        
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
        # Fallback to dummy data
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
class Particle:
    def __init__(self, start_pos, end_pos, speed=0.1):
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.pos = np.array(start_pos)
        self.speed = speed
        self.progress = 0.0
        self.finished = False

    def update(self):
        self.progress += self.speed
        if self.progress >= 1.0:
            self.progress = 1.0
            self.finished = True
        self.pos = self.start_pos + (self.end_pos - self.start_pos) * self.progress

def visualize_network(X, y, model):
    input_size = X.shape[1]
    
    # Create Graph
    G = nx.DiGraph()
    layers = [input_size, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE]
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
    
    pos = {}
    
    for i, layer_size in enumerate(layers):
        x = i * 4
        y_offset = (layer_size - 1) / 2.0
        for j in range(layer_size):
            node_id = f"{i}_{j}"
            G.add_node(node_id, layer=i)
            pos[node_id] = (x, j - y_offset)
    
    # Setup Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('#0a0a0a')
    
    # Set limits to ensure labels fit
    ax.set_xlim(-1, len(layers) * 4 + 1)
    max_y = max(layers) / 2.0 + 1
    ax.set_ylim(-max_y, max_y)
    
    # Draw static edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#333333', alpha=0.2, width=1)
    
    nodes_draw = nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#111111', 
                                       node_size=400, edgecolors='#444444', linewidths=1)
    
    # Labels for layers
    for i, name in enumerate(layer_names):
        ax.text(i * 4, -max(layers)/2 - 1.5, name, ha='center', color='cyan', 
                fontsize=14, fontweight='bold', alpha=0.8)

    # Particle management
    particles = []
    
    # Labels for activations
    labels = []
    for node_id, (px, py) in pos.items():
        t = ax.text(px, py + 0.4, '', ha='center', color='white', fontsize=8, alpha=0)
        labels.append((node_id, t))

    # Animation state
    current_frame_data = {'idx': 0, 'substep': 0}
    particle_scatter = ax.scatter([], [], c='yellow', s=20, zorder=5, edgecolors='white', linewidths=0.5)

    def update(frame):
        SUBSTEPS_PER_ROW = 15
        row_idx = current_frame_data['idx']
        substep = current_frame_data['substep']
        
        if substep == 0:
            input_row = torch.FloatTensor(X[row_idx]).unsqueeze(0)
            _ = model(input_row)
            
            # Emit particles: Input -> Hidden 1
            for j in range(layers[0]):
                for k in range(layers[1]):
                    u, v = f"0_{j}", f"1_{k}"
                    particles.append(Particle(pos[u], pos[v], speed=1.0/SUBSTEPS_PER_ROW))
                    
        elif substep == int(SUBSTEPS_PER_ROW * 0.3):
            # Emit: Hidden 1 -> Hidden 2
            for j in range(layers[1]):
                for k in range(layers[2]):
                    u, v = f"1_{j}", f"2_{k}"
                    particles.append(Particle(pos[u], pos[v], speed=1.0/SUBSTEPS_PER_ROW))
                    
        elif substep == int(SUBSTEPS_PER_ROW * 0.6):
            # Emit: Hidden 2 -> Output
            for j in range(layers[2]):
                for k in range(layers[3]):
                    u, v = f"2_{j}", f"3_{k}"
                    particles.append(Particle(pos[u], pos[v], speed=1.0/SUBSTEPS_PER_ROW))

        # Update particles
        for p in particles[:]:
            p.update()
            if p.finished:
                particles.remove(p)
        
        # Update node visuals
        act_vals = list(torch.FloatTensor(X[row_idx]).numpy().flatten())
        act_vals.extend(model.activations['fc1'].detach().numpy().flatten())
        act_vals.extend(model.activations['fc2'].detach().numpy().flatten())
        act_vals.extend(model.activations['out'].detach().numpy().flatten())
        
        colors = []
        sizes = []
        for i, val in enumerate(act_vals):
            intensity = 1 / (1 + np.exp(-val))
            colors.append((0, intensity, intensity))
            sizes.append(400 + intensity * 600)
            
            node_id = list(pos.keys())[i]
            for nid, txt in labels:
                if nid == node_id:
                    txt.set_text(f"{val:.2f}")
                    txt.set_alpha(intensity)
        
        nodes_draw.set_color(colors)
        nodes_draw.set_sizes(sizes)
        
        # Update particles scatter
        if particles:
            pxs = [p.pos[0] for p in particles]
            pys = [p.pos[1] for p in particles]
            particle_scatter.set_offsets(np.c_[pxs, pys])
        else:
            particle_scatter.set_offsets(np.empty((0, 2)))

        # Advance
        current_frame_data['substep'] += 1
        if current_frame_data['substep'] >= SUBSTEPS_PER_ROW:
            current_frame_data['substep'] = 0
            current_frame_data['idx'] = (current_frame_data['idx'] + 1) % len(X)
        
        ax.set_title(f"Data Flow | Step: {current_frame_data['idx']} | Pred: {['Down', 'Neutral', 'Up'][np.argmax(act_vals[-3:])]}", 
                     color='white', fontsize=16)
        
        return nodes_draw, particle_scatter,

    ax.axis('off')
    # Adjust top margin to make room for title
    plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)
    
    print("Opening live visualization window...")
    ani = animation.FuncAnimation(fig, update, frames=len(X)*15, interval=30, blit=False)
    plt.show()
    
    print("Saving a snippet to 'results/nn_activation.gif'...")
    try:
        # Create a separate short animation for saving
        save_ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=False)
        save_ani.save('results/nn_activation.gif', writer='pillow', fps=20)
        print("Animation snippet saved.")
    except Exception as e:
        print(f"Could not save GIF: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Neural Network Visualization")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (e.g., AAPL, MSFT)")
    args = parser.parse_args()

    X, y = load_and_process_data(args.ticker)
    print(f"Data shape: {X.shape}")
    
    # Initialize Model
    model = StockPredictor(X.shape[1], HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE)
    
    # Visualize
    visualize_network(X, y, model)