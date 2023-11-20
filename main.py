import praw
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import config

# Download the VADER lexicon for sentiment analysis
nltk.download("vader_lexicon")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Reddit credentials
reddit = praw.Reddit(
    client_id=config.client_id,
    client_secret=config.client_secret,
    user_agent=config.user_agent,
)

# Define a list of stock symbols that you're interested in
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Subreddit selection
subreddit_name = "finance"

# Data collection and sentiment analysis
stock_sentiments = {stock: [] for stock in stocks}
subreddit = reddit.subreddit(subreddit_name)

for stock in stocks:
    for post in subreddit.search(stock, limit=50):  # limit can be adjusted
        sentiment_score = sia.polarity_scores(post.title)["compound"]
        stock_sentiments[stock].append((post.created_utc, sentiment_score))

# Data Visualization
plt.figure(figsize=(10, 6))

for stock, data in stock_sentiments.items():
    if data:
        dates = [datetime.fromtimestamp(time, tz=pytz.utc) for time, _ in data]
        sentiments = [sentiment for _, sentiment in data]
        plt.plot(dates, sentiments, marker="o", label=stock)

plt.title("Sentiment Analysis of Stock-related Reddit Posts")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.legend()
plt.show()
