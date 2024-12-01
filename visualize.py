import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

def visualize_sample(model_gru, model_lstm, model_mamba, dataset, index=0):
    """Visualize input sequence and predictions for a single sample"""
    model_gru.eval()
    model_lstm.eval()
    model_mamba.eval()

    x, y_true = dataset[index]
    
    # Add batch dimension for model input
    x_batch = x.unsqueeze(0)
    with torch.no_grad():
        y_pred_gru = model_gru(x_batch).squeeze()
        y_pred_lstm = model_lstm(x_batch).squeeze()
        y_pred_mamba = model_mamba(x_batch).squeeze()
    
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
    
    y_pred_gru_denorm = dataset.scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_pred_gru), 3)),
                       y_pred_gru.numpy().reshape(-1, 1),
                       np.zeros((len(y_pred_gru), 1))], axis=1)
    )[:, 3]

    y_pred_lstm_denorm = dataset.scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_pred_lstm), 3)),
                       y_pred_lstm.numpy().reshape(-1, 1),
                       np.zeros((len(y_pred_lstm), 1))], axis=1)
    )[:, 3] 
    
    y_pred_mamba_denorm = dataset.scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_pred_mamba), 3)),
                       y_pred_mamba.numpy().reshape(-1, 1),
                       np.zeros((len(y_pred_mamba), 1))], axis=1)
    )[:, 3] 
    
    # Plot predictions
    ax_pred.plot(y_true_denorm, label='True Close Price')
    ax_pred.plot(y_pred_gru_denorm, label='Predicted Close Price GRU')
    ax_pred.plot(y_pred_lstm_denorm, label='Predicted Close Price LSTM')
    ax_pred.plot(y_pred_mamba_denorm, label='Predicted Close Price Mamba')
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
    print(f"\nPrediction Error Metrics GRU:")
    mse = ((y_true_denorm - y_pred_gru_denorm) ** 2).mean()
    mae = abs(y_true_denorm - y_pred_gru_denorm).mean()
    print(f"MSE: ${mse:.2f}")
    print(f"MAE: ${mae:.2f}")

    print(f"\nPrediction Error Metrics LSTM:")
    mse = ((y_true_denorm - y_pred_lstm_denorm) ** 2).mean()
    mae = abs(y_true_denorm - y_pred_lstm_denorm).mean()
    print(f"MSE: ${mse:.2f}")
    print(f"MAE: ${mae:.2f}")       

    print(f"\nPrediction Error Metrics Mamba:")
    mse = ((y_true_denorm - y_pred_mamba_denorm) ** 2).mean()
    mae = abs(y_true_denorm - y_pred_mamba_denorm).mean()
    print(f"MSE: ${mse:.2f}")
    print(f"MAE: ${mae:.2f}")

# Add this to the main() function after model training
def visualize_results(model_gru, model_lstm, model_mamba, train_dataset, val_dataset):
    print("\nVisualizing training sample:")
    visualize_sample(model_gru, model_lstm, model_mamba, train_dataset)
    
    print("\nVisualizing validation sample:")
    visualize_sample(model_gru, model_lstm, model_mamba, val_dataset)