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
from visualize import visualize_results
from models_pytorch.minGRU import StockPredictorGRU
from models_pytorch.minLSTM import StockPredictorLSTM
from models_pytorch.mamba import StockPredictorMamba

# Constants
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
SEQ_LEN = 60  # Number of time steps to look back
INPUT_DIM = 5  # OHLCV features
HIDDEN_DIM = 64
NUM_EPOCHS = 10
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


# log-space version of minLSTM

           
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
    model_gru = StockPredictorGRU(INPUT_DIM, HIDDEN_DIM, expansion_factor=1.5)
    optimizer_gru = Adam(model_gru.parameters(), lr=LEARNING_RATE)
    criterion_gru = nn.MSELoss()

    model_lstm = StockPredictorLSTM(INPUT_DIM, HIDDEN_DIM)
    optimizer_lstm = Adam(model_lstm.parameters(), lr=LEARNING_RATE)
    criterion_lstm = nn.MSELoss()
    
    model_mamba = StockPredictorMamba(INPUT_DIM, HIDDEN_DIM)
    optimizer_mamba = Adam(model_mamba.parameters(), lr=LEARNING_RATE)
    criterion_mamba = nn.MSELoss()
    
    # Training loop
    best_val_loss_gru = float('inf')
    best_val_loss_lstm = float('inf')
    best_val_loss_mamba = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model_gru.train()
        model_lstm.train()
        model_mamba.train()

        train_loss_gru = 0
        train_loss_lstm = 0
        train_loss_mamba = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for x, y in pbar:
            optimizer_gru.zero_grad()
            optimizer_lstm.zero_grad()
            optimizer_mamba.zero_grad()

            pred_gru = model_gru(x)
            pred_lstm = model_lstm(x)
            pred_mamba = model_mamba(x)

            loss_gru = criterion_gru(pred_gru, y)
            loss_lstm = criterion_lstm(pred_lstm, y)
            loss_mamba = criterion_mamba(pred_mamba, y)

            loss_gru.backward()
            loss_lstm.backward()
            loss_mamba.backward()

            optimizer_gru.step()
            optimizer_lstm.step()
            optimizer_mamba.step()

            train_loss_gru += loss_gru.item()
            train_loss_lstm += loss_lstm.item()
            train_loss_mamba += loss_mamba.item()
            
            wandb.log({"train_loss_gru": loss_gru.item(), "train_loss_lstm": loss_lstm.item(), "train_loss_mamba": loss_mamba.item()})
        
        # Validation
        model_gru.eval()
        model_lstm.eval()
        model_mamba.eval()

        val_loss_gru = 0
        val_loss_lstm = 0
        val_loss_mamba = 0

        with torch.no_grad():
            for x, y in val_loader:

                pred_gru = model_gru(x)
                pred_lstm = model_lstm(x)
                pred_mamba = model_mamba(x)

                loss_gru = criterion_gru(pred_gru, y)
                loss_lstm = criterion_lstm(pred_lstm, y)
                loss_mamba = criterion_mamba(pred_mamba, y)

                val_loss_gru += loss_gru.item()
                val_loss_lstm += loss_lstm.item()
                val_loss_mamba += loss_mamba.item()
        
        val_loss_gru /= len(val_loader)
        val_loss_lstm /= len(val_loader)
        val_loss_mamba /= len(val_loader)
        
        wandb.log({
            "epoch": epoch,
            "validation_loss_gru": val_loss_gru,
            "validation_loss_lstm": val_loss_lstm,
            "validation_loss_mamba": val_loss_mamba
        })
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print(f'Training Loss GRU: {train_loss_gru/len(train_loader):.4f}')
        print(f'Training Loss LSTM: {train_loss_lstm/len(train_loader):.4f}')
        print(f'Training Loss Mamba: {train_loss_mamba/len(train_loader):.4f}')

        print(f'Validation Loss GRU: {val_loss_gru:.4f}')
        print(f'Validation Loss LSTM: {val_loss_lstm:.4f}')
        print(f'Validation Loss Mamba: {val_loss_mamba:.4f}')
        
        # Save best model
        if val_loss_gru < best_val_loss_gru:
            best_val_loss_gru = val_loss_gru
            torch.save(model_gru.state_dict(), f'best_model_gru_{int(time.time())}.pt')
        
        if val_loss_lstm < best_val_loss_lstm:
            best_val_loss_lstm = val_loss_lstm
            torch.save(model_lstm.state_dict(), f'best_model_lstm_{int(time.time())}.pt')

        if val_loss_mamba < best_val_loss_mamba:
            best_val_loss_mamba = val_loss_mamba
            torch.save(model_mamba.state_dict(), f'best_model_mamba_{int(time.time())}.pt')

    wandb.finish()

    visualize_results(model_gru, model_lstm, model_mamba, train_dataset, val_dataset)

if __name__ == "__main__":
    main()
