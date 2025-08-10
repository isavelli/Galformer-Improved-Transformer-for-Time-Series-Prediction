# GalformerV2_Advanced.py: An enhanced Transformer model for multi-feature, multi-step stock market forecasting.

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
import glob
import math
import os
import yfinance as yf


# --- Helper Functions (Integrated for self-containment) ---

def get_angles(pos, k, d: int):
    """Calculate angles for positional encoding."""
    i = k // 2
    angles = pos / (10000 ** (2 * i / d))
    return angles


def positional_encoding(positions: int, d_model: int, device: torch.device):
    """Precomputes a matrix with all the positional encodings."""
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = torch.from_numpy(angle_rads).float().unsqueeze(0).to(device)
    return pos_encoding


def create_causal_mask(size: int, device: torch.device):
    """Creates a causal mask for the decoder's self-attention."""
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


# --- Configuration ---
class G:
    """A static class for holding hyperparameters."""
    # Data Preprocessing
    batch_size = 64
    src_len = 30  # Input sequence length
    tgt_len = 5  # Target sequence length (for multi-step prediction)
    # Model Architecture
    num_features = 3  # Uses Close Change, Volume Change, and Volatility
    d_model = 512
    dense_dim = 2048
    num_heads = 8
    num_layers = 6
    dropout_rate = 0.1
    # Training
    epochs = 150
    learning_rate = 0.0005
    # Early Stopping
    early_stopping_patience = 10
    # LR Scheduler
    lr_scheduler_step_size = 5
    lr_scheduler_gamma = 0.9


# --- PyTorch Model Definition ---

class FullyConnected(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model, dense_dim):
        super(FullyConnected, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dense_dim),
            nn.GELU(),
            nn.LayerNorm(dense_dim),
            nn.Linear(dense_dim, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """A single layer of the Transformer Encoder."""

    def __init__(self, d_model, num_heads, dense_dim, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = FullyConnected(d_model, dense_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


class Encoder(nn.Module):
    """The Transformer Encoder stack."""

    def __init__(self, num_layers, d_model, num_heads, dense_dim, dropout_rate, max_pos_encoding, device):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Linear(G.num_features, d_model)
        self.pos_encoding = positional_encoding(max_pos_encoding, d_model, device)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dense_dim, dropout_rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        seq_len = x.shape[1]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
        return x


class DecoderLayer(nn.Module):
    """A single layer of the Transformer Decoder."""

    def __init__(self, d_model, num_heads, dense_dim, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.self_mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True)
        self.cross_mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = FullyConnected(d_model, dense_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, causal_mask, padding_mask):
        attn_output1, _ = self.self_mha(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout1(attn_output1))
        attn_output2, _ = self.cross_mha(x, enc_output, enc_output, attn_mask=padding_mask)
        x = self.norm2(x + self.dropout2(attn_output2))
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x


class Decoder(nn.Module):
    """The Transformer Decoder stack."""

    def __init__(self, num_layers, d_model, num_heads, dense_dim, dropout_rate, max_pos_encoding, device):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Linear(G.num_features, d_model)
        self.pos_encoding = positional_encoding(max_pos_encoding, d_model, device)
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dense_dim, dropout_rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, causal_mask):
        seq_len = x.shape[1]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, causal_mask, None)
        return x


class Transformer(nn.Module):
    """The complete Transformer model."""

    def __init__(self, device):
        super(Transformer, self).__init__()
        self.encoder = Encoder(G.num_layers, G.d_model, G.num_heads, G.dense_dim, G.dropout_rate, G.src_len, device)
        self.decoder = Decoder(G.num_layers, G.d_model, G.num_heads, G.dense_dim, G.dropout_rate, G.tgt_len, device)
        self.final_layer = nn.Linear(G.d_model, G.num_features)

    def forward(self, src, tgt):
        causal_mask = create_causal_mask(tgt.shape[1], src.device)
        enc_output = self.encoder(src, None)
        dec_output = self.decoder(tgt, enc_output, causal_mask)
        final_output = self.final_layer(dec_output)
        return final_output


# --- Data Loading and Processing ---
def get_price_data(filename):
    """
    Loads and preprocesses multiple features from a CSV file.
    Features: Close price change, Volume change, High-Low volatility.
    """
    df = pd.read_csv(filename)

    # === FIX APPLIED HERE ===
    # Explicitly convert price/volume columns to numeric types.
    # This prevents the TypeError if pandas misinterprets the CSV.
    numeric_cols = ['High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # === END OF FIX ===

    # Ensure dataframe is sorted by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Feature 1: Daily close price change
    df['Close_Change'] = df['Close'].diff(1)

    # Feature 2: Daily volume change
    df['Volume_Change'] = df['Volume'].diff(1)

    # Feature 3: Intra-day volatility
    df['Volatility'] = df['High'] - df['Low']

    # Drop rows with NaN values created by .diff() or conversion errors
    df = df.dropna().reset_index(drop=True)

    return df[['Close_Change', 'Volume_Change', 'Volatility']]


def load_data(df, src_len, tgt_len, normalize=True):
    """Splits data, creates sequences, and normalizes for multi-feature input."""
    data = df.values
    division_rate1 = 0.7
    division_rate2 = 0.85
    row1 = round(division_rate1 * data.shape[0])
    row2 = round(division_rate2 * data.shape[0])

    train = data[:int(row1), :]
    valid = data[int(row1):int(row2), :]
    test = data[int(row2):, :]

    scaler = preprocessing.StandardScaler()
    if normalize:
        train = scaler.fit_transform(train)
        valid = scaler.transform(valid)
        test = scaler.transform(test)

    def create_dataset(dataset, src_len, tgt_len):
        X, y = [], []
        num_samples = dataset.shape[0] - src_len - tgt_len + 1
        for i in range(num_samples):
            X.append(dataset[i:i + src_len])
            y.append(dataset[i + src_len:i + src_len + tgt_len])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train, src_len, tgt_len)
    X_valid, y_valid = create_dataset(valid, src_len, tgt_len)
    X_test, y_test = create_dataset(test, src_len, tgt_len)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler


# --- Loss, Metrics, and Training ---

def calculate_metrics(pred, real, feature_index=0):
    """
    Calculates key metrics for a specific feature (default is the first one, e.g., Close_Change).
    """
    pred_np = pred.cpu().detach().numpy()[:, :, feature_index].flatten()
    real_np = real.cpu().detach().numpy()[:, :, feature_index].flatten()

    MAPE = sklearn.metrics.mean_absolute_percentage_error(real_np, pred_np)
    RMSE = np.sqrt(sklearn.metrics.mean_squared_error(real_np, pred_np))
    MAE = sklearn.metrics.mean_absolute_error(real_np, pred_np)
    R2 = r2_score(real_np, pred_np)

    metrics = {'MAPE': MAPE, 'RMSE': RMSE, 'MAE': MAE, 'R2': R2}
    print('Evaluation Metrics (for the primary feature):\n', pd.DataFrame(metrics, index=[0]))
    return metrics


# --- Main Execution Block ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define data folder and file path
    data_folder = 'data/indices'
    os.makedirs(data_folder, exist_ok=True)
    gspc_file_path = os.path.join(data_folder, '^GSPC.csv')

    # --- Automatic Data Download ---
    # If the data file doesn't exist, download it from Yahoo Finance
    if not os.path.exists(gspc_file_path):
        print(f"Data file not found. Downloading S&P 500 data to {gspc_file_path}...")
        try:
            # Download historical data for the S&P 500
            # auto_adjust=True handles splits/dividends and silences the warning
            gspc_data = yf.download(
                '^GSPC',
                start='1990-01-01',
                end=pd.to_datetime('today').strftime('%Y-%m-%d'),
                auto_adjust=True
            )
            if gspc_data.empty:
                raise ValueError("No data downloaded. Check ticker and network connection.")
            gspc_data.reset_index(inplace=True)  # Move 'Date' from index to a column
            gspc_data.to_csv(gspc_file_path, index=False)
            print("Data downloaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to download data: {e}")
            exit()

    all_files = glob.glob(gspc_file_path)

    if not all_files:
        print(f"Error: Data file is still not available at {gspc_file_path}.")

    for filename in all_files:
        print(f"\n--- Processing {filename} ---")

        # 1. Load and Process Data
        df = get_price_data(filename)
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler = load_data(df, G.src_len, G.tgt_len)

        # Convert to PyTorch Tensors
        X_train_t = torch.from_numpy(X_train).float().to(device)
        y_train_t = torch.from_numpy(y_train).float().to(device)
        X_valid_t = torch.from_numpy(X_valid).float().to(device)
        y_valid_t = torch.from_numpy(y_valid).float().to(device)
        X_test_t = torch.from_numpy(X_test).float().to(device)
        y_test_t = torch.from_numpy(y_test).float().to(device)

        # Create DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=G.batch_size, shuffle=True)
        valid_dataset = TensorDataset(X_valid_t, y_valid_t)
        valid_loader = DataLoader(valid_dataset, batch_size=G.batch_size)

        # 2. Initialize Model, Optimizer, Loss, and Schedulers
        model = Transformer(device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=G.learning_rate)
        criterion = nn.MSELoss()
        scheduler = StepLR(optimizer, step_size=G.lr_scheduler_step_size, gamma=G.lr_scheduler_gamma)

        # 3. Training Loop with Early Stopping
        print("Starting training...")
        best_valid_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(G.epochs):
            model.train()
            total_train_loss = 0
            for src, tgt in train_loader:
                dec_input_teacher_forcing = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)

                optimizer.zero_grad()
                output = model(src, dec_input_teacher_forcing)
                loss = criterion(output, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            model.eval()
            total_valid_loss = 0
            with torch.no_grad():
                for src, tgt in valid_loader:
                    dec_input_teacher_forcing = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
                    output = model(src, dec_input_teacher_forcing)
                    loss = criterion(output, tgt)
                    total_valid_loss += loss.item()
            avg_valid_loss = total_valid_loss / len(valid_loader)

            scheduler.step()

            print(
                f'Epoch {epoch + 1}/{G.epochs} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}')

            # Early stopping logic
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
                print(f"Validation loss improved. Saving model state.")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= G.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        # Load the best model state before evaluation
        if best_model_state:
            model.load_state_dict(best_model_state)
            print("\nLoaded best model state for evaluation.")

        # 4. Evaluation (Autoregressive Inference)
        print("\nStarting evaluation...")
        model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(len(X_test_t)):
                src_sample = X_test_t[i:i + 1]
                dec_input_sample = src_sample[:, -1:, :]

                for _ in range(G.tgt_len):
                    output = model(src_sample, dec_input_sample)
                    last_pred_step = output[:, -1:, :]
                    dec_input_sample = torch.cat([dec_input_sample, last_pred_step], dim=1)

                final_prediction = dec_input_sample[:, 1:, :]
                predictions.append(final_prediction)

        predictions_t = torch.cat(predictions, dim=0)

        # Denormalize predictions and actual values
        predictions_scaled = scaler.inverse_transform(predictions_t.cpu().numpy().reshape(-1, G.num_features))
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, G.num_features))

        # Reshape back to (num_samples, tgt_len, num_features)
        predictions_scaled = predictions_scaled.reshape(-1, G.tgt_len, G.num_features)
        y_test_scaled = y_test_scaled.reshape(-1, G.tgt_len, G.num_features)

        calculate_metrics(torch.tensor(predictions_scaled), torch.tensor(y_test_scaled), feature_index=0)

        # 5. Plotting (the first step of the multi-step prediction)
        plt.figure(figsize=(15, 7))
        plt.plot(y_test_scaled[:, 0, 0], color='black', label='Real Stock Price Change (1st day)')
        plt.plot(predictions_scaled[:, 0, 0], color='green', label='Predicted Stock Price Change (1st day)', alpha=0.7)
        plt.title(f'1-Step Ahead Prediction for {os.path.basename(filename)}', fontsize=20)
        plt.xlabel('Time')
        plt.ylabel('Price Change')
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()