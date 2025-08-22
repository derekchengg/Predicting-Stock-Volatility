import pandas as pd
import yfinance as yf
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import time

# reddit api scrape
# reference: https://medium.com/geekculture/a-complete-guide-to-web-scraping-reddit-with-python-16e292317a52
reddit = praw.Reddit(
    client_id="placeholder",
    client_secret="placeholder",
    user_agent="testagent/0.1"
)

# analyze comments
# reference: https://medium.com/@rslavanyageetha/vader-a-comprehensive-guide-to-sentiment-analysis-in-python-c4f1868b0d2e
# reference: https://github.com/cjhutto/vaderSentiment
analyzer = SentimentIntensityAnalyzer()

wsb = reddit.subreddit('wallstreetbets')

def main():
    # reference: https://ranaroussi.github.io/yfinance/reference/index.html
    # getting $GME data from yfinance
    print("Grabbing $GME data")
    gme = yf.download('GME', start='2021-01-01', end='2021-03-31', auto_adjust=True)
    gme = gme.reset_index()
    gme['ticker'] = 'GME'
    gme.to_csv('scrape_stock/stock_data_GME.csv', index=False)
    print("finished $GME data")

    # getting $OPEN data from yfinance
    print("Grabbing $OPEN data")
    open_stock = yf.download('OPEN', start='2025-05-15', end='2025-08-10', auto_adjust=True)
    open_stock = open_stock.reset_index()
    open_stock['ticker'] = 'OPEN'
    open_stock.to_csv('scrape_stock/stock_data_OPEN.csv', index=False)
    print("finished $OPEN data")

    # stores Reddit data
    all_data = []
    
    print("GME Reddit Scrape")
    count = 0
    # reference: https://praw.readthedocs.io/en/stable/code_overview/models/subreddit.html#praw.models.Subreddit.search
    # set to time_filter='all because GME was popular in 2021 and having same parameter as a recent ticker was not scraping data properly
    for submission in wsb.search('GME', time_filter='all', limit=150):
        created = datetime.fromtimestamp(submission.created_utc)
        # needs extra layer filter due to time_filter='all'
        if created.year == 2021 and created.month <= 3:
            sentiment = analyzer.polarity_scores(submission.title)
            all_data.append({
                'date': created.strftime('%Y-%m-%d'),
                'ticker': 'GME',
                'type': 'post',
                'score': submission.score,
                'num_comments': submission.num_comments,
                'sentiment': sentiment['compound']
            })
            
            # getting top 3 comments
            submission.comments.replace_more(limit=0)
            for comment in list(submission.comments)[:3]:
                if hasattr(comment, 'body'):
                    # max 500 characters to reduce scraping time
                    comment_sentiment = analyzer.polarity_scores(comment.body[:500])
                    all_data.append({
                        'date': created.strftime('%Y-%m-%d'),
                        'ticker': 'GME',
                        'type': 'comment',
                        'score': comment.score,
                        'num_comments': 0,
                        'sentiment': comment_sentiment['compound']
                    })
            
            count += 1
            time.sleep(0.2)
    print("GME scrape finished.")
    
    # OPEN data from recent posts
    print("OPEN Reddit Scrape")
    count = 0
    # set to time_filter='month' because OPEN is more recent stock and anything other than this filter also wouldn't scrape data properly
    for submission in wsb.search('OPEN', time_filter='month', limit=150):
        created = datetime.fromtimestamp(submission.created_utc)
        sentiment = analyzer.polarity_scores(submission.title)
        all_data.append({
            'date': created.strftime('%Y-%m-%d'),
            'ticker': 'OPEN',
            'type': 'post',
            'score': submission.score,
            'num_comments': submission.num_comments,
            'sentiment': sentiment['compound']
        })

        # getting top 3 comments
        submission.comments.replace_more(limit=0)
        for comment in list(submission.comments)[:3]:
            if hasattr(comment, 'body'):
                # max 500 characters to reduce scraping time
                comment_sentiment = analyzer.polarity_scores(comment.body[:500])
                all_data.append({
                    'date': created.strftime('%Y-%m-%d'),
                    'ticker': 'OPEN',
                    'type': 'comment',
                    'score': comment.score,
                    'num_comments': 0,
                    'sentiment': comment_sentiment['compound']
                })
        
        count += 1
        time.sleep(0.2)
    print("OPEN scrape finished.")

    # all the data scraped in one file
    df = pd.DataFrame(all_data)
    df.to_csv('scrape_stock/reddit_raw.csv', index=False)
    print("raw file data completed")
    
    # since there is more than one posts a day when it is scraped it will group the day to daily so it's easier to read
    daily = df.groupby(['date', 'ticker']).agg({
        'sentiment': ['mean', 'count']
    }).reset_index()
    daily.columns = ['date', 'ticker', 'avg_sentiment', 'mention_count']
    daily.to_csv('scrape_stock/reddit_daily.csv', index=False)
    print("daily file data completed")

if __name__ == '__main__':
    main()