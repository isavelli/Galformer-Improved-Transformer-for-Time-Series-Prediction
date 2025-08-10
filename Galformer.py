# Galformer_Classifier_Enhanced_V2.py: A Transformer model with an expanded set of technical indicators.

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import yfinance as yf


# --- Helper Functions ---

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


# --- Configuration ---
class G:
    """Static class for hyperparameters."""
    # Data Preprocessing
    batch_size = 64
    seq_len = 30
    # Model Architecture
    num_features = 11  # CHANGED: Increased from 7 to 11 for new indicators
    d_model = 128
    dense_dim = 512
    num_heads = 8
    num_layers = 4
    dropout_rate = 0.1
    # Training
    epochs = 50
    learning_rate = 0.0001
    weight_decay = 1e-5
    # Early Stopping
    early_stopping_patience = 10
    # LR Scheduler
    lr_scheduler_patience = 3
    lr_scheduler_factor = 0.5


# --- PyTorch Model Definition ---

class FullyConnected(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model, dense_dim):
        super(FullyConnected, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dense_dim),
            nn.GELU(),
            nn.Linear(dense_dim, d_model)
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


class DirectionalTransformer(nn.Module):
    """The complete Transformer-based Classifier."""

    def __init__(self, device):
        super(DirectionalTransformer, self).__init__()
        self.encoder = Encoder(G.num_layers, G.d_model, G.num_heads, G.dense_dim, G.dropout_rate, G.seq_len, device)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classification_head = nn.Sequential(
            nn.Linear(G.d_model, G.d_model // 2),
            nn.GELU(),
            nn.Dropout(G.dropout_rate),
            nn.Linear(G.d_model // 2, 2)
        )

    def forward(self, src):
        enc_output = self.encoder(src, None)
        pooled_output = self.pooling(enc_output.permute(0, 2, 1)).squeeze(-1)
        final_output = self.classification_head(pooled_output)
        return final_output


# --- Data Loading and Processing ---
def get_price_data(filename):
    """
    CHANGED: Added Stochastic Oscillator, ATR, and OBV indicators.
    """
    df = pd.read_csv(filename)
    numeric_cols = ['High', 'Low', 'Close', 'Open', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Add technical indicators using pandas_ta
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.stoch(k=14, d=3, append=True)  # ADDED: Stochastic Oscillator
    df.ta.atr(length=14, append=True)  # ADDED: Average True Range
    df.ta.obv(append=True)  # ADDED: On-Balance Volume

    # Original features
    df['Close_Change'] = df['Close'].pct_change(fill_method=None)
    df['Volume_Change'] = df['Volume'].pct_change(fill_method=None)
    df['Volatility'] = (df['High'] - df['Low']) / df['Close']

    # Replace any infinite values that resulted from division by zero with NaN.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Create the binary target variable
    df['Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df = df.dropna().reset_index(drop=True)

    # Define the expanded list of feature columns
    feature_cols = [
        'Close_Change', 'Volume_Change', 'Volatility',
        'RSI_14', 'MACD_12_26_9', 'BBP_20_2.0',
        'STOCHk_14_3_3', 'STOCHd_14_3_3',  # Stochastic
        'ATRr_14',  # ATR
        'OBV'  # On-Balance Volume
    ]
    # We will exclude BBL_20_2.0 as BBP_20_2.0 (Bollinger Band Percentage) already captures the info.
    return df[feature_cols + ['Direction']]


def load_data(df, seq_len, normalize=True):
    """Splits data, creates sequences, and normalizes for classification."""

    feature_names = [col for col in df.columns if col != 'Direction']
    features = df[feature_names].values
    target = df['Direction'].values

    division_rate1 = 0.7
    division_rate2 = 0.85
    row1 = round(division_rate1 * features.shape[0])
    row2 = round(division_rate2 * features.shape[0])

    train_features = features[:int(row1), :]
    valid_features = features[int(row1):int(row2), :]
    test_features = features[int(row2):, :]

    train_target = target[:int(row1)]
    valid_target = target[int(row1):int(row2)]
    test_target = target[int(row2):]

    scaler = preprocessing.StandardScaler()
    if normalize:
        train_features = scaler.fit_transform(train_features)
        valid_features = scaler.transform(valid_features)
        test_features = scaler.transform(test_features)

    def create_dataset(features, target, seq_len):
        X, y = [], []
        num_samples = features.shape[0] - seq_len
        for i in range(num_samples):
            X.append(features[i:i + seq_len])
            y.append(target[i + seq_len - 1])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_features, train_target, seq_len)
    X_valid, y_valid = create_dataset(valid_features, valid_target, seq_len)
    X_test, y_test = create_dataset(test_features, test_target, seq_len)

    class_counts = np.bincount(y_train)
    class_weights = torch.tensor([len(y_train) / c for c in class_counts], dtype=torch.float)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler, class_weights


# --- Metrics and Visualization ---
def calculate_and_plot_metrics(pred_logits, real_labels, title="Confusion Matrix"):
    """Calculates classification metrics and plots a confusion matrix."""
    pred_probs = torch.softmax(pred_logits, dim=1)
    predictions = torch.argmax(pred_probs, dim=1).cpu().numpy()
    real_labels = real_labels.cpu().numpy()

    accuracy = accuracy_score(real_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(real_labels, predictions, average='binary',
                                                               zero_division=0)

    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
    print('Evaluation Metrics:\n', pd.DataFrame(metrics, index=[0]))

    cm = confusion_matrix(real_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'])
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Direction')
    plt.ylabel('Actual Direction')
    plt.show()

    return metrics


# --- Main Execution Block ---
if __name__ == '__main__':
    # You may need to install pandas_ta: pip install pandas_ta
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_folder = 'data/indices'
    os.makedirs(data_folder, exist_ok=True)
    gspc_file_path = os.path.join(data_folder, '^GSPC.csv')

    if not os.path.exists(gspc_file_path):
        print(f"Downloading S&P 500 data to {gspc_file_path}...")
        try:
            gspc_data = yf.download('^GSPC', start='2000-01-01', end=pd.to_datetime('today').strftime('%Y-%m-%d'))
            if gspc_data.empty: raise ValueError("No data downloaded.")
            gspc_data.reset_index(inplace=True)
            gspc_data.to_csv(gspc_file_path, index=False)
            print("Data downloaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to download data: {e}");
            exit()

    print(f"\n--- Processing {gspc_file_path} ---")

    # 1. Load and Process Data
    df = get_price_data(gspc_file_path)
    X_train, y_train, X_valid, y_valid, X_test, y_test, scaler, class_weights = load_data(df, G.seq_len)

    class_weights = class_weights.to(device)
    print(f"Using {len(df.columns) - 1} features.")
    print(f"Calculated class weights: {class_weights.cpu().numpy()}")

    # Convert to PyTorch Tensors
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_valid_t = torch.from_numpy(X_valid).float().to(device)
    y_valid_t = torch.from_numpy(y_valid).long().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=G.batch_size, shuffle=True)
    valid_dataset = TensorDataset(X_valid_t, y_valid_t)
    valid_loader = DataLoader(valid_dataset, batch_size=G.batch_size)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=G.batch_size)

    # 2. Initialize Model, Optimizer, Loss, and Scheduler
    model = DirectionalTransformer(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=G.learning_rate, weight_decay=G.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=G.lr_scheduler_patience, factor=G.lr_scheduler_factor)

    # 3. Training Loop with Early Stopping
    print("Starting training...")
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(G.epochs):
        model.train()
        total_train_loss = 0
        for src, tgt in train_loader:
            optimizer.zero_grad()
            output_logits = model(src)
            loss = criterion(output_logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for src, tgt in valid_loader:
                output_logits = model(src)
                loss = criterion(output_logits, tgt)
                total_valid_loss += loss.item()
        avg_valid_loss = total_valid_loss / len(valid_loader)

        scheduler.step(avg_valid_loss)
        print(
            f'Epoch {epoch + 1}/{G.epochs} | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f} | LR: {optimizer.param_groups[0]["lr"]:.7f}')

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= G.early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("\nLoaded best model state for evaluation.")

    # 4. Final Evaluation on Test Set
    print("\nStarting final evaluation on test data...")
    model.eval()
    all_preds = []
    all_reals = []
    with torch.no_grad():
        for src, tgt in test_loader:
            output_logits = model(src)
            all_preds.append(output_logits)
            all_reals.append(tgt)

    all_preds_t = torch.cat(all_preds, dim=0)
    all_reals_t = torch.cat(all_reals, dim=0)

    calculate_and_plot_metrics(all_preds_t, all_reals_t, title="Final Test Set Confusion Matrix")