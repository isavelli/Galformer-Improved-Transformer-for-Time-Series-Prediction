import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
import glob
import math
from math import floor

# Import helper functions from the new PyTorch helper file
from transformer_helper_pytorch import positional_encoding, create_causal_mask


# --- Configuration ---
class G:
    """A static class for holding hyperparameters."""
    # Data Preprocessing
    batch_size = 64
    src_len = 20
    dec_len = 1
    tgt_len = 1
    window_size = src_len
    mulpr_len = tgt_len
    # Network Architecture
    d_model = 512
    dense_dim = 2048
    num_features = 1
    num_heads = 8
    num_layers = 6
    dropout_rate = 0.1
    # Training
    epochs = 100  # Reduced for quicker demonstration
    learning_rate = 0.001


# --- PyTorch Model Definition ---

class FullyConnected(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model, dense_dim):
        super(FullyConnected, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dense_dim),
            nn.ReLU(),
            nn.BatchNorm1d(dense_dim),
            nn.Linear(dense_dim, d_model),
            nn.BatchNorm1d(d_model)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, d_model)
        # BatchNorm1d expects (batch_size, channels, seq_len)
        x_permuted = x.permute(0, 2, 1)
        out_permuted = self.net(x_permuted)
        out = out_permuted.permute(0, 2, 1)
        return out


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
        # Multi-Head Attention
        attn_output, _ = self.mha(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed Forward Network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


class Encoder(nn.Module):
    """The Transformer Encoder stack."""

    def __init__(self, num_layers, d_model, num_heads, dense_dim, dropout_rate, max_pos_encoding, device):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.device = device

        self.embedding = nn.Linear(G.num_features, d_model)
        self.pos_encoding = positional_encoding(max_pos_encoding, d_model, device)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dense_dim, dropout_rate) for _ in range(num_layers)
        ])
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
        # Masked Multi-Head Self-Attention
        attn_output1, _ = self.self_mha(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout1(attn_output1))

        # Cross-Attention
        attn_output2, _ = self.cross_mha(x, enc_output, enc_output, attn_mask=padding_mask)
        x = self.norm2(x + self.dropout2(attn_output2))

        # Feed Forward Network
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x


class Decoder(nn.Module):
    """The Transformer Decoder stack."""

    def __init__(self, num_layers, d_model, num_heads, dense_dim, dropout_rate, max_pos_encoding, device):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.device = device

        self.embedding = nn.Linear(G.num_features, d_model)
        self.pos_encoding = positional_encoding(max_pos_encoding, d_model, device)

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dense_dim, dropout_rate) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, causal_mask):
        seq_len = x.shape[1]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, causal_mask, None)  # No padding mask for now
        return x


class Transformer(nn.Module):
    """The complete Transformer model."""

    def __init__(self, device, num_layers=G.num_layers, d_model=G.d_model, num_heads=G.num_heads, dense_dim=G.dense_dim,
                 dropout_rate=G.dropout_rate):
        super(Transformer, self).__init__()
        self.device = device
        self.encoder = Encoder(num_layers, d_model, num_heads, dense_dim, dropout_rate, G.src_len, device)
        self.decoder = Decoder(num_layers, d_model, num_heads, dense_dim, dropout_rate, G.dec_len, device)
        self.final_layer = nn.Linear(d_model, G.num_features)

    def forward(self, src, tgt):
        causal_mask = create_causal_mask(tgt.shape[1], self.device)
        enc_output = self.encoder(src, None)  # No padding mask for now
        dec_output = self.decoder(tgt, enc_output, causal_mask)
        final_output = self.final_layer(dec_output)
        return final_output


# --- Data Loading and Processing ---

def get_stock_data(filename):
    """Loads and preprocesses stock data from a CSV file."""
    df = pd.read_csv(filename)
    df['Adj Close'] = df['Close']
    df.drop(['Open', 'High', 'Low', 'Volume', 'Date'], axis=1, inplace=True, errors='ignore')

    # Calculate difference
    diff_values = df['Adj Close'].diff(1).dropna().values
    df = df.iloc[1:].copy()
    df['Adj Close'] = diff_values

    return df


def load_data(df, seq_len, mul, normalize=True):
    """Splits data into train/valid/test sets and creates sequences."""
    data = df.values
    division_rate1 = 0.6
    division_rate2 = 0.8
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

    def create_dataset(dataset, seq_len, tgt_len, mul):
        X, y = [], []
        num_samples = dataset.shape[0] - seq_len - mul + 1
        for i in range(0, num_samples, mul):
            X.append(dataset[i:i + seq_len])
            y.append(dataset[i + seq_len:i + seq_len + tgt_len])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train, seq_len, G.tgt_len, mul)
    X_valid, y_valid = create_dataset(valid, seq_len, G.tgt_len, mul)
    X_test, y_test = create_dataset(test, seq_len, G.tgt_len, mul)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler


# --- Loss, Metrics, and Training ---

def up_down_accuracy_loss(pred, real):
    """Custom loss combining MSE and directional accuracy."""
    mse = torch.mean(torch.square(pred - real))

    accu = real * pred
    accu = torch.relu(accu)
    accu = torch.sign(accu)
    accu = torch.mean(accu)

    # Combine MSE and directional accuracy
    loss = mse + (1 - accu) * 0.1  # Weighting factor for directional loss
    return loss


def calculate_metrics(pred, real):
    """Calculates MAPE, RMSE, MAE, and R2 score."""
    pred_np = pred.cpu().detach().numpy().flatten()
    real_np = real.cpu().detach().numpy().flatten()

    MAPE = sklearn.metrics.mean_absolute_percentage_error(real_np, pred_np)
    RMSE = np.sqrt(sklearn.metrics.mean_squared_error(real_np, pred_np))
    MAE = sklearn.metrics.mean_absolute_error(real_np, pred_np)
    R2 = r2_score(real_np, pred_np)

    metrics = {'MAPE': MAPE, 'RMSE': RMSE, 'MAE': MAE, 'R2': R2}
    print('Evaluation Metrics:\n', pd.DataFrame(metrics, index=[0]))
    return metrics


# --- Main Execution Block ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Example: Use a placeholder for data files if not present
    try:
        all_files = [x for x in glob.glob('data/dax40/*.csv', recursive=False)] + \
                    [x for x in glob.glob('data/indices/^GDAXI.csv', recursive=False)]
        if not all_files: raise FileNotFoundError
    except FileNotFoundError:
        print("Data files not found. Please place stock data in './data/...'")
        # Create a dummy file for demonstration if none exist
        dummy_data = {'Date': pd.to_datetime(pd.date_range(start='1/1/2010', periods=1000)),
                      'Close': np.random.rand(1000) * 100 + 500}
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv('dummy_stock.csv', index=False)
        all_files = ['dummy_stock.csv']

    for filename in all_files:
        print(f"\n--- Processing {filename} ---")

        # 1. Load Data
        df = get_stock_data(filename)
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler = load_data(df, G.src_len, G.mulpr_len)

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

        # 2. Initialize Model, Optimizer, Loss
        model = Transformer(device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=G.learning_rate)
        criterion = up_down_accuracy_loss

        # 3. Training Loop
        print("Starting training...")
        for epoch in range(G.epochs):
            model.train()
            total_train_loss = 0
            for src, tgt in train_loader:
                # The target for the decoder input is the same as the output in this setup
                # We use the source as a proxy for the decoder input for the first step
                dec_input = src[:, -G.dec_len:, :]

                optimizer.zero_grad()
                output = model(src, dec_input)
                loss = criterion(output, tgt)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            model.eval()
            total_valid_loss = 0
            with torch.no_grad():
                for src, tgt in valid_loader:
                    dec_input = src[:, -G.dec_len:, :]
                    output = model(src, dec_input)
                    loss = criterion(output, tgt)
                    total_valid_loss += loss.item()

            avg_valid_loss = total_valid_loss / len(valid_loader)

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch {epoch + 1}/{G.epochs}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}')

        # 4. Evaluation
        print("\nStarting evaluation...")
        model.eval()
        with torch.no_grad():
            dec_input_test = X_test_t[:, -G.dec_len:, :]
            predictions_t = model(X_test_t, dec_input_test)

        # Denormalize predictions and actual values
        predictions_scaled = scaler.inverse_transform(predictions_t.cpu().numpy().reshape(-1, G.num_features))
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, G.num_features))

        # Calculate metrics on the difference values
        calculate_metrics(torch.tensor(predictions_scaled), torch.tensor(y_test_scaled))

        # 5. Plotting
        plt.figure(figsize=(15, 7))
        plt.plot(y_test_scaled.flatten(), color='black', label='Real Stock Price Change')
        plt.plot(predictions_scaled.flatten(), color='green', label='Predicted Stock Price Change', alpha=0.7)
        plt.title(f'Stock Price Change Prediction for {filename.split("/")[-1]}', fontsize=20)
        plt.xlabel('Time')
        plt.ylabel('Price Change')
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

