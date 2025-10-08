import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from matplotlib.ticker import MultipleLocator

def load_individual_stock_data():
    """
    Loads stock data from individual CSV files (e.g., 'AAPL_stock_data.csv')
    in the 'setup_packages' folder and returns a dictionary of DataFrames,
    where each key is the ticker symbol.
    """
    # Define the tickers you want to load
    ticker_list = ['AAPL', 'PG', 'JNJ', 'JPM']
    
    # The folder where your CSV files are stored
    data_folder = "setup_packages"

    # A dictionary to hold each ticker's DataFrame
    stock_dataframes = {}

    print("Loading data from local CSV files...")

    # Loop through each ticker to read its corresponding CSV file
    for ticker in ticker_list:
        # Construct the full path to the CSV file using the correct naming convention
        file_path = os.path.join(data_folder, f"{ticker}_stock_data.csv")
        
        # Check if the file exists before trying to read it
        if os.path.exists(file_path):
            try:
                # Read the CSV file, skipping the second and third rows which may contain metadata.
                # The index is set to the first column (Date) and parsed as dates.
                company_df = pd.read_csv(file_path, index_col=0, parse_dates=True, skiprows=[1, 2])
                
                # Set a proper name for the index column.
                company_df.index.name = 'Date'
                
                # Store the DataFrame in the dictionary with the ticker as the key
                stock_dataframes[ticker] = company_df
                print(f" - Successfully loaded {ticker} data.")
            except Exception as e:
                print(f" - Error processing file for {ticker}: {e}")

    if not stock_dataframes:
        print("\nNo data was loaded.")
        return None

    print("\nSuccessfully loaded all individual stock data.")
    return stock_dataframes

if __name__ == "__main__":
    all_stocks = load_individual_stock_data()

    if all_stocks:
        AAPL = all_stocks['AAPL']
        
        # ----------LSTM Model for Stock Price Prediction----------
        
        # --- Simplified Data Preparation ---
        # Use only the 'Close' price for a cleaner model
        data = AAPL[['Close']].values
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # --- Create sequences of 60 days of data and the next day's price ---
        time_steps = 60
        X, y = [], []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i-time_steps:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        # Reshape data for LSTM [samples, time_steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # --- Time Series Cross-Validation (4 Folds) ---
        tscv = TimeSeriesSplit(n_splits=4)

        mse_scores, rmse_scores, mae_scores, r2_scores = [], [], [], []

        # Create 2x2 grid for plots BEFORE the loop
        fig_loss, axes_loss = plt.subplots(2, 2, figsize=(16, 12))
        fig_preds, axes_preds = plt.subplots(2, 2, figsize=(16, 12))
        
        axes_loss = axes_loss.flatten()
        axes_preds = axes_preds.flatten()
        
        fig_loss.suptitle('Model Training & Validation Loss (Simplified Model)', fontsize=16)
        fig_preds.suptitle('Stock Price Prediction (Simplified Model)', fontsize=16)

        for fold, (train_index, val_index) in enumerate(tscv.split(X)):
            print(f"\nTraining fold {fold + 1}...")
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Build a simpler model for the cleaner data
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            model.compile(optimizer='adam', loss='mean_squared_error')

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
            
            # --- Evaluation ---
            val_predictions = model.predict(X_val)
            val_predictions_inv = scaler.inverse_transform(val_predictions)
            y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1))

            mse = mean_squared_error(y_val_inv, val_predictions_inv)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_inv, val_predictions_inv)
            r2 = r2_score(y_val_inv, val_predictions_inv)

            mse_scores.append(mse)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)

            print(f"\nFold {fold + 1} Model Evaluation Metrics:")
            print(f" - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

            # --- Plotting for the current fold ---
            ax_loss = axes_loss[fold]
            ax_loss.plot(history.history['loss'], label='Training Loss')
            ax_loss.plot(history.history['val_loss'], label='Validation Loss')
            ax_loss.set_title(f'Fold {fold + 1}')
            ax_loss.legend()
            ax_loss.grid(True)
            ax_loss.xaxis.set_major_locator(MultipleLocator(10))

            ax_pred = axes_preds[fold]
            ax_pred.plot(y_val_inv, color='blue', label='Actual Stock Price')
            ax_pred.plot(val_predictions_inv, color='red', label='Predicted Stock Price')
            ax_pred.set_title(f'Fold {fold + 1}')
            ax_pred.legend()
            ax_pred.grid(True)
            ax_pred.xaxis.set_major_locator(MultipleLocator(20))

        for i in range(2):
            axes_loss[i + 2].set_xlabel('Epochs')
            axes_preds[i + 2].set_xlabel('Time in Days')

        fig_loss.tight_layout(rect=[0, 0.03, 1, 0.96])
        fig_preds.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        plt.show()

        print(f"\n---------- Cross-Validation Summary ----------")
        print(f"Mean MSE: {np.mean(mse_scores):.4f}")
        print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
        print(f"Mean MAE: {np.mean(mae_scores):.4f}")
        print(f"Mean R2: {np.mean(r2_scores):.4f}")
