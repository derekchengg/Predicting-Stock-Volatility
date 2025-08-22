# Predicting Stock Volatility Using Reddit Sentiment Analysis

This project analyzes whether Reddit sentiment which is analyzed from VADER sentiment API to see whether a post or comment is positive or negative from r/wallstreetbets and using that information as well as activity (posts/comments) on a stock if it can predict stock volatility. The analysis compares GameStop (GME) during the 2021 meme stock phenomenon with Opendoor (OPEN) which is a rising meme stock in the past couple of months. Using this sentiment on older data and newer data.

## Required Libraries
- pip install pandas
- pip install yfinance
- pip install praw
- pip install vaderSentiment
- pip install matplotlib
- pip install scipy
- pip install scikit-learn
- pip install statsmodels

## Additional Note:
If you do want to run the scrape yourself, you will need to:
- Go to https://www.reddit.com/prefs/apps
- Create a new application (script type)
- Update the credentials in scrape.py:

`reddit = praw.Reddit(
    client_id="your_client_id",
    client_secret="your_client_secret", 
    user_agent="your_app_name/1.0"
)`

## How to Run:
1) python scrape.py (already ran so the files are in "scrape" folder)
2) python merge_clean.py
3) python statistical_tests.py
4) python train.py
5) python visuals.py

## APIs Used:
- https://praw.readthedocs.io/
- https://github.com/cjhutto/vaderSentiment
- https://ranaroussi.github.io/yfinance/
