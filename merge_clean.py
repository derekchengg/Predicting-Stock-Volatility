import pandas as pd
import numpy as np
import os

def main():
    os.makedirs('merge_cleaned', exist_ok=True)
    
    # load stock prices from yfinance and reddit data
    gme_prices = pd.read_csv('scrape_stock/stock_data_GME.csv')
    open_prices = pd.read_csv('scrape_stock/stock_data_OPEN.csv')
    reddit = pd.read_csv('scrape_stock/reddit_daily.csv')

    # combining stock prices into a single DataFrame
    prices = pd.concat([gme_prices, open_prices])
    # setting each column to the correct type
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices['Close'] = pd.to_numeric(prices['Close'], errors='coerce')
    prices['Volume'] = pd.to_numeric(prices['Volume'], errors='coerce')
    # remove any duplicate rows
    prices = prices.drop_duplicates(subset=['Date', 'ticker'])
    # sort the prices by ticker and then date
    prices = prices.sort_values(['ticker', 'Date'])
    # calculate daily returns for volatility calculation
    prices['ret'] = prices.groupby('ticker')['Close'].pct_change()
    # calculate rolling volatility
    prices['vol_5d'] = prices.groupby('ticker')['ret'].transform(lambda x: x.rolling(5).std())

    # setting date column for reddit data
    reddit['date'] = pd.to_datetime(reddit['date'])
    # merge reddit data with stock data
    dataset = pd.merge(prices, reddit, left_on=['Date', 'ticker'], right_on=['date', 'ticker'], how='left')
    # fill missing reddit data with zeros
    reddit_cols = ['avg_sentiment', 'mention_count']
    dataset[reddit_cols] = dataset[reddit_cols].fillna(0)
    # create high volatility indicator (above 75th percentile)
    dataset['high_vol'] = dataset.groupby('ticker')['vol_5d'].transform(lambda x: (x > x.quantile(0.75)).astype(int))
    # add features used in modeling
    dataset['mentioned_on_reddit'] = (dataset['mention_count'] > 0).astype(int)
    # select final columns (only what's actually used)
    final_cols = ['Date', 'ticker', 'Close', 'Volume', 'vol_5d', 'high_vol', 'mentioned_on_reddit', 'mention_count', 'avg_sentiment']
    dataset = dataset[final_cols].dropna(subset=['vol_5d'])
    dataset.to_csv('merge_cleaned/dataset.csv', index=False)
    
    # show correlation preview for both tickers
    for ticker in ['GME', 'OPEN']:
        ticker_data = dataset[dataset['ticker'] == ticker]
        if len(ticker_data) > 0:
            # show basic stats first
            reddit_days = ticker_data['mentioned_on_reddit'].sum()
            print(f'\n{ticker} stats:')
            print(f'Total days checked: {len(ticker_data)}')
            print(f'# of days mentioned: {reddit_days} ({reddit_days/len(ticker_data)*100:.1f}%)')
            print(f'Avg mentions on days ticker was hot: {ticker_data[ticker_data["mention_count"] > 0]["mention_count"].mean():.1f}' if reddit_days > 0 else '  Avg mentions: N/A')
            print(f'High volatility days: {ticker_data["high_vol"].sum()} ({ticker_data["high_vol"].mean()*100:.1f}%)')

            # only calculate correlations if there's variance
            if ticker_data["mention_count"].std() > 0 and ticker_data["vol_5d"].std() > 0:
                mention_corr = ticker_data["mention_count"].corr(ticker_data["vol_5d"])
                print(f'mention_count vs vol_5d correlation: {mention_corr:.3f}')
            else:
                print(f'mention_count vs vol_5d correlation: N/A (no variance)')
            
            if ticker_data["mentioned_on_reddit"].std() > 0 and ticker_data["high_vol"].std() > 0:
                reddit_corr = ticker_data["mentioned_on_reddit"].corr(ticker_data["high_vol"])
                print(f'mentioned_on_reddit vs high_vol correlation: {reddit_corr:.3f}')
            else:
                print(f'mentioned_on_reddit vs high_vol correlation: N/A (no variance)')

if __name__ == '__main__':
    main()