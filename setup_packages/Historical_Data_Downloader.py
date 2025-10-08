# This script downloads the last 3 years of stock data for a list of companies
# from Yahoo Finance and saves each to a local CSV file.

# First, you need to make sure you have the required libraries installed.
# You can install them using pip:
# pip install yfinance pandas

import yfinance as yf
import pandas as pd
from datetime import date, timedelta

def download_stock_data(ticker_list):
    """
    Downloads the last 3 years of stock data for each ticker in the list
    and saves it to a unique CSV file.
    """
    # Calculate the start and end dates
    end_date = date.today()
    start_date = end_date - timedelta(days=3*365)

    for ticker_symbol in ticker_list:
        try:
            print(f"Downloading data for {ticker_symbol} from {start_date} to {end_date}...")

            # Download the stock data using yfinance
            # The data will include columns like Open, High, Low, Close, Adj Close, and Volume
            stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

            if stock_data.empty:
                print(f"No data found for {ticker_symbol}. Please check the ticker symbol and date range.")
                continue # Skip to the next ticker

            # Define the local file path to store the data
            file_path = f"{ticker_symbol}_stock_data.csv"

            # Save the DataFrame to a CSV file
            stock_data.to_csv(file_path)

            print(f"\nSuccessfully downloaded data for {ticker_symbol} and saved it to '{file_path}'")
            print(f"\nHere's a preview of the first 5 rows for {ticker_symbol}:")
            print(stock_data.head())
            print("-" * 50) # Separator for clarity

        except Exception as e:
            print(f"An error occurred while downloading data for {ticker_symbol}: {e}")
            print("-" * 50)


if __name__ == "__main__":
    # Define the list of ticker symbols for non-tech companies
    # PG: Procter & Gamble (Consumer Goods)
    # JNJ: Johnson & Johnson (Healthcare)
    # JPM: JPMorgan Chase & Co. (Finance)
    stocks_to_download = ['AAPL','PG', 'JNJ', 'JPM']
    download_stock_data(stocks_to_download)