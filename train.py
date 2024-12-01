import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import wandb
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf

def visualize_sample(model, model_lstm, dataset, index=0):
    """Visualize input sequence and predictions for a single sample"""
    model.eval()
    model_lstm.eval()
    x, y_true = dataset[index]
    
    # Add batch dimension for model input
    x_batch = x.unsqueeze(0)
    with torch.no_grad():
        y_pred = model(x_batch).squeeze()
        y_pred_lstm = model_lstm(x_batch).squeeze()
    
    # Denormalize the data
    ohlcv_data = dataset.scaler.inverse_transform(x.numpy())
    
    # Create DataFrame for mplfinance
    df = pd.DataFrame(
        ohlcv_data,
        columns=['Open', 'High', 'Low', 'Close', 'Volume']
    )
    df.index = pd.date_range(end='2023', periods=len(df), freq='D')
    
    # Create candlestick chart
    # fig_candle = mpf.figure(figsize=(15, 5))
    # ax_candle = fig_candle.add_subplot(1,1,1)
    mpf.plot(
        df,
        type='candle',
        style='charles',
        title='Input Sequence Features (OHLCV)',
        volume=True,
        figsize=(15, 5),
        returnfig=True
    )
    
    # Create prediction plot
    fig_pred, ax_pred = plt.subplots(figsize=(15, 5))
    
    # Denormalize predictions and true values
    y_true_denorm = dataset.scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_true), 3)),
                       y_true.numpy().reshape(-1, 1),
                       np.zeros((len(y_true), 1))], axis=1)
    )[:, 3]
    
    y_pred_denorm = dataset.scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_pred), 3)),
                       y_pred.numpy().reshape(-1, 1),
                       np.zeros((len(y_pred), 1))], axis=1)
    )[:, 3]

    y_pred_lstm_denorm = dataset.scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_pred_lstm), 3)),
                       y_pred_lstm.numpy().reshape(-1, 1),
                       np.zeros((len(y_pred_lstm), 1))], axis=1)
    )[:, 3]
    
    # Plot predictions
    ax_pred.plot(y_true_denorm, label='True Close Price')
    ax_pred.plot(y_pred_denorm, label='Predicted Close Price')
    ax_pred.plot(y_pred_lstm_denorm, label='Predicted Close Price LSTM')
    ax_pred.set_title('Price Predictions vs True Values')
    ax_pred.set_ylabel('Price ($)')
    ax_pred.legend()
    ax_pred.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\nSample Statistics:")
    print(f"Input sequence length: {len(x)}")
    print(f"Number of features: {x.shape[1]}")
    print(f"Close price range: ${y_true_denorm.min():.2f} - ${y_true_denorm.max():.2f}")
    print(f"\nPrediction Error Metrics:")
    mse = ((y_true_denorm - y_pred_denorm) ** 2).mean()
    mae = abs(y_true_denorm - y_pred_denorm).mean()
    print(f"MSE: ${mse:.2f}")
    print(f"MAE: ${mae:.2f}")


# Add this to the main() function after model training
def visualize_results(model, model_lstm, train_dataset, val_dataset):
    print("\nVisualizing training sample:")
    visualize_sample(model, model_lstm, train_dataset)
    
    print("\nVisualizing validation sample:")
    visualize_sample(model, model_lstm, val_dataset)
# Constants
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
SEQ_LEN = 60  # Number of time steps to look back
INPUT_DIM = 5  # OHLCV features
HIDDEN_DIM = 64
NUM_EPOCHS = 5
SEED = 42

torch.manual_seed(SEED)

class StockDataset(Dataset):
    def __init__(self, data, seq_len, scaler=None):
        if scaler is None:
            self.scaler = MinMaxScaler()
            self.data = self.scaler.fit_transform(data)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(data)
        
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]
        y = self.data[index + 1:index + self.seq_len + 1, 3]  # Next close prices
        return torch.FloatTensor(x), torch.FloatTensor(y)


def heinsen_associative_scan_log(log_coeffs, log_values):
    """
    Implement the heinsen associative scan in log-space for numerical stability.
    This is the same implementation as in the original minGRU.
    """
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

def exists(val):
    return val is not None

# Simplified minGRU for stock prediction
class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, expansion_factor=1.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Like minGRU, use expansion factor for inner dimension
        dim_inner = int(hidden_dim * expansion_factor)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Combined hidden and gate computation like minGRU
        self.to_hidden_and_gate = nn.Linear(hidden_dim, dim_inner * 2, bias=False)
        
        # Output projection with expansion handling
        self.to_out = nn.Linear(dim_inner, hidden_dim, bias=False) if expansion_factor != 1. else nn.Identity()
        
        # Final projection for stock prediction
        self.final_proj = nn.Linear(hidden_dim, 1, bias=False)

    def g(self, x):
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    def log_g(self, x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        batch_size, seq_len, _ = x.shape

        # Project input to hidden dimension
        x = self.input_proj(x)
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        if seq_len == 1:
            # Sequential processing
            hidden = self.g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)
        else:
            # Parallel processing
            log_coeffs = -F.softplus(gate)
            log_z = -F.softplus(-gate)
            log_tilde_h = self.log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((self.log_g(prev_hidden), log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]
        out = self.to_out(out)
        
        # Final projection for stock price
        predictions = self.final_proj(out)
        predictions = predictions.squeeze(-1)

        if not return_next_prev_hidden:
            return predictions

        return predictions, next_prev_hidden


# log-space version of minLSTM

class minLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False):
        super(minLSTM, self).__init__()
        self.hidden_size = hidden_size

        # Linear layers for forget gate, input gate, and candidate hidden state
        self.linear_f = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_i = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h = nn.Linear(input_size, hidden_size, bias=bias)

        # Activation functions
    def g(self, x):
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    def log_g(self, x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def parallel_scan_log(self, log_coeffs, log_values):
        """
        Implements the parallel_scan_log function as described.

        Args:
            log_coeffs (Tensor): Logarithm of forget gates, shape (batch_size, seq_len, hidden_size).
            log_values (Tensor): Concatenated tensor of log_h0 and (log_i + log_tilde_h),
                                 shape (batch_size, seq_len + 1, hidden_size).

        Returns:
            Tensor: The hidden states after parallel scan, shape (batch_size, seq_len, hidden_size).
        """
        # Compute a_star: cumulative sum of log_coeffs along the sequence dimension (dim=1)
        # Shape: (batch_size, seq_len, hidden_size)
        a_star = torch.cumsum(log_coeffs, dim=1)

        # Pad a_star with a zero at the beginning of the sequence dimension
        # Shape after padding: (batch_size, seq_len + 1, hidden_size)
        a_star_padded = F.pad(a_star, (0, 0, 1, 0), "constant", 0)

        # Compute log_h0_plus_b_star using logcumsumexp for numerical stability
        # log_values has shape: (batch_size, seq_len + 1, hidden_size)
        # a_star_padded has shape: (batch_size, seq_len + 1, hidden_size)
        # log_values - a_star_padded computes element-wise subtraction
        # logcumsumexp is applied along the sequence dimension (dim=1)
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star_padded, dim=1)
        # Compute log_h by adding a_star_padded and log_h0_plus_b_star
        log_h = a_star_padded + log_h0_plus_b_star  # Shape: (batch_size, seq_len +1, hidden_size)

        # Remove the first element corresponding to the padding
        # Shape: (batch_size, seq_len, hidden_size)
        log_h = log_h[:, 1:, :]

        # Exponentiate to get h in the original space
        h = torch.exp(log_h)

        return h  # Shape: (batch_size, seq_len, hidden_size)
    
    def forward(self, x, h_0):
        """
        Forward pass of the minLSTM.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size).
            h_0 (Tensor): Initial hidden state of shape (batch_size, hidden_size).

        Returns:
            Tensor: The hidden state after processing, shape depends on the mode:
                    - Sequential Mode: (batch_size, hidden_size)
                    - Parallel Mode: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, input_size = x.size()

        if seq_len == 1:
            # === Sequential Mode ===
            # Extract the single time step
            x_t = x[:, 0, :]  # Shape: (batch_size, input_size)
            h_prev = h_0      # Shape: (batch_size, hidden_size)

            # Compute gates
            f_t = torch.sigmoid(self.linear_f(x_t))  # Forget gate
            i_t = torch.sigmoid(self.linear_i(x_t))  # Input gate

            # Compute candidate hidden state
            tilde_h_t = self.g(self.linear_h(x_t))   # Candidate hidden state

            # Normalize gates
            f_prime_t = f_t / (f_t + i_t)
            i_prime_t = i_t / (f_t + i_t)

            # Update hidden state
            h_t = f_prime_t * h_prev + i_prime_t * tilde_h_t  # Shape: (batch_size, hidden_size)

            return h_t  # Sequential output
        else:
            # === Parallel Mode ===
            # Compute log-space forget and input gates
            log_f = -F.softplus(-self.linear_f(x))  # Shape: (batch_size, seq_len, hidden_size)
            log_i = -F.softplus(-self.linear_i(x))  # Shape: (batch_size, seq_len, hidden_size)

            # Compute log-space candidate hidden state
            log_tilde_h = self.log_g(self.linear_h(x))  # Shape: (batch_size, seq_len, hidden_size)

            # Compute difference for gate normalization
            diff = F.softplus(-self.linear_f(x)) - F.softplus(-self.linear_i(x))  # Shape: (batch_size, seq_len, hidden_size)

            # Update log gates based on the difference
            log_f = -F.softplus(diff)      # Updated forget gate
            log_i = -F.softplus(-diff)     # Updated input gate

            # Compute log-space initial hidden state
            log_h0 = self.log_g(h_0)       # Shape: (batch_size, hidden_size)

            # Prepare concatenated tensor for parallel scan
            log_h0 = log_h0.unsqueeze(1)    # Shape: (batch_size, 1, hidden_size)
            log_i_plus_log_tilde_h = log_i + log_tilde_h  # Shape: (batch_size, seq_len, hidden_size)
            concatenated = torch.cat([log_h0, log_i_plus_log_tilde_h], dim=1)  # Shape: (batch_size, seq_len + 1, hidden_size)

            # Perform parallel scan
            h = self.parallel_scan_log(log_f, concatenated)  # Shape: (batch_size, seq_len, hidden_size)

            return h  # Parallel output

class StockPredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # minLSTM layer
        self.lstm = minLSTM(hidden_dim, hidden_dim)
        
        # Final projection for stock prediction
        self.final_proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, h_0=None):
        batch_size, seq_len, _ = x.shape
        
        # Initialize h_0 if not provided
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Project input to hidden dimension
        x = self.input_proj(x)
        
        # Process through minLSTM
        hidden_states = self.lstm(x, h_0)
        
        # Final projection for stock price
        predictions = self.final_proj(hidden_states)
        predictions = predictions.squeeze(-1)

        return predictions
           
def prepare_stock_data(file_path):
    """Prepare stock data from CSV file"""
    df = pd.read_csv(file_path)
    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    return train_data, val_data

def main():
    # Initialize wandb
    wandb.init(
        project="stock-prediction-minGRU",
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seq_len": SEQ_LEN,
            "hidden_dim": HIDDEN_DIM,
        }
    )
    
    # Prepare data
    train_data, val_data = prepare_stock_data('AABA_2006-01-01_to_2018-01-01.csv')
    
    # Create datasets and dataloaders
    train_dataset = StockDataset(train_data, SEQ_LEN)
    val_dataset = StockDataset(val_data, SEQ_LEN, train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = StockPredictor(INPUT_DIM, HIDDEN_DIM, expansion_factor=1.5)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    model_lstm = StockPredictorLSTM(INPUT_DIM, HIDDEN_DIM)
    optimizer_lstm = Adam(model_lstm.parameters(), lr=LEARNING_RATE)
    criterion_lstm = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    best_val_loss_lstm = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        model_lstm.train()
        train_loss = 0
        train_loss_lstm = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for x, y in pbar:
            optimizer.zero_grad()
            optimizer_lstm.zero_grad()
            pred = model(x)
            pred_lstm = model_lstm(x)
            loss = criterion(pred, y)
            loss_lstm = criterion_lstm(pred_lstm, y)
            loss.backward()
            loss_lstm.backward()
            optimizer.step()
            optimizer_lstm.step()
            train_loss += loss.item()
            train_loss_lstm += loss_lstm.item()
            
            wandb.log({"train_loss": loss.item(), "train_loss_lstm": loss_lstm.item()})
        
        # Validation
        model.eval()
        model_lstm.eval()
        val_loss = 0
        val_loss_lstm = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                pred_lstm = model_lstm(x)
                loss = criterion(pred, y)
                loss_lstm = criterion_lstm(pred_lstm, y)
                val_loss += loss.item()
                val_loss_lstm += loss_lstm.item()
        
        val_loss /= len(val_loader)
        val_loss_lstm /= len(val_loader)
        wandb.log({
            "epoch": epoch,
            "validation_loss": val_loss,
            "validation_loss_lstm": val_loss_lstm
        })
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Training Loss LSTM: {train_loss_lstm/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Loss LSTM: {val_loss_lstm:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_{int(time.time())}.pt')
        
        if val_loss_lstm < best_val_loss_lstm:
            best_val_loss_lstm = val_loss_lstm
            torch.save(model_lstm.state_dict(), f'best_model_lstm_{int(time.time())}.pt')

    wandb.finish()

    visualize_results(model, model_lstm, train_dataset, val_dataset)

if __name__ == "__main__":
    main()
