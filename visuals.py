import pandas as pd
import matplotlib.pyplot as plt
import os

def create_simple_plots():
    os.makedirs('visuals', exist_ok=True)
    dataset = pd.read_csv('merge_cleaned/dataset.csv')
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    
    # plot 1: time series
    # reference: https://machinelearningmastery.com/time-series-data-visualization-with-python/
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # GME plot
    gme_data = dataset[dataset['ticker'] == 'GME']
    ax1.plot(gme_data['Date'], gme_data['vol_5d'], color='blue')
    ax1.set_title('GME: Volatility Over Time')
    ax1.set_ylabel('Volatility')
    
    # OPEN plot
    open_data = dataset[dataset['ticker'] == 'OPEN']
    ax2.plot(open_data['Date'], open_data['vol_5d'], color='orange')
    ax2.set_title('OPEN: Volatility Over Time')
    ax2.set_ylabel('Volatility')
    ax2.set_xlabel('Date')
    plt.savefig('visuals/timeseries.png')
    plt.show()
    
    # plot 2: box plot
    # reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html
    plt.figure(figsize=(8, 6))
    none_vol = []
    low_vol = []
    medium_vol = []
    high_vol = []
    
    for i in range(len(dataset)):
        mentions = dataset.iloc[i]['mention_count']
        vol = dataset.iloc[i]['vol_5d']
        
        if pd.isna(vol):
            continue
            
        if mentions == 0:
            none_vol.append(vol)
        elif mentions <= 5:
            low_vol.append(vol)
        elif mentions <= 20:
            medium_vol.append(vol)
        else:
            high_vol.append(vol)
            
    plt.boxplot([none_vol, low_vol, medium_vol, high_vol], labels=['None', 'Low', 'Medium', 'High'])
    plt.title('Volatility by Reddit Activity Level')
    plt.xlabel('Activity Level')
    plt.ylabel('Volatility')
    plt.savefig('visuals/boxplot.png')
    plt.show()
    
    # plot 3: scatter plot
    # reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    plt.figure(figsize=(10, 6))
    active_data = dataset[dataset['mention_count'] > 0]
    gme_active = active_data[active_data['ticker'] == 'GME']
    plt.scatter(gme_active['avg_sentiment'], gme_active['vol_5d'], label='GME', color='blue')
    open_active = active_data[active_data['ticker'] == 'OPEN']
    plt.scatter(open_active['avg_sentiment'], open_active['vol_5d'], label='OPEN', color='orange')
    plt.title('Sentiment vs Volatility')
    plt.xlabel('Average Sentiment')
    plt.ylabel('Volatility')
    plt.legend()
    plt.savefig('visuals/scatterplot.png')
    plt.show()
    
    # plot 4: bar chart
    # reference: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
    plt.figure(figsize=(8, 6))
    gme_corr = gme_data['mention_count'].corr(gme_data['vol_5d'])
    open_corr = open_data['mention_count'].corr(open_data['vol_5d'])
    stocks = ['GME', 'OPEN']
    correlations = [gme_corr, open_corr]
    plt.bar(stocks, correlations, color=['blue', 'orange'])
    plt.title('Correlation: Reddit Mentions vs Volatility')
    plt.ylabel('Correlation')
    plt.ylim(0, 1)
    plt.text(0, gme_corr + 0.02, f'{gme_corr:.3f}', ha='center')
    plt.text(1, open_corr + 0.02, f'{open_corr:.3f}', ha='center')
    plt.savefig('visuals/barchart.png')
    plt.show()

if __name__ == '__main__':
    create_simple_plots()