import praw
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy import create_engine
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from langchain_community.chat_models import ChatOpenAI  # ✅ Fixed import

# ✅ Reddit API credentials
CLIENT_ID = "Fj0Sk2BLVJ0PCBA_r41zEg"
CLIENT_SECRET = "arhY7vtO1j-XiMsyyiwl4kKWI_PLeQ"
USER_AGENT = "python:RedditScraper:v1.0 (by u/Monk481)"
OPENAI_API_KEY = "sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # Masked key for security

# ✅ Initialize Reddit API
reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)

# ✅ Connect to SQLite
conn = sqlite3.connect("reddit_analysis.db", check_same_thread=False)
cursor = conn.cursor()

# ✅ Create Database Table (if not exists)
cursor.execute("""
CREATE TABLE IF NOT EXISTS reddit_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    subreddit TEXT,
    score INTEGER,
    content TEXT,
    url TEXT,
    created_at TEXT,
    sentiment TEXT
)
""")
conn.commit()

# 🎯 **Function to Scrape Reddit Data**
def store_reddit_data(years=2):
    """Fetch all posts & comments from the user 'opinionsareus' within the last X years."""
    username = "opinionsareus"  # ✅ Fixed username
    user = reddit.redditor(username)
    data = []
    cutoff_time = datetime.utcnow() - timedelta(days=years * 365)

    for submission in user.submissions.new(limit=None):  
        post_time = datetime.utcfromtimestamp(submission.created_utc)
        if post_time < cutoff_time:
            break
        data.append(("Post", submission.subreddit.display_name, submission.score, 
                     submission.title, submission.url, post_time, None))

    for comment in user.comments.new(limit=None):  
        comment_time = datetime.utcfromtimestamp(comment.created_utc)
        if comment_time < cutoff_time:
            break
        data.append(("Comment", comment.subreddit.display_name, comment.score, 
                     comment.body, f"https://www.reddit.com{comment.permalink}", comment_time, None))

    cursor.executemany("INSERT INTO reddit_data (type, subreddit, score, content, url, created_at, sentiment) VALUES (?, ?, ?, ?, ?, ?, ?)", data)
    conn.commit()
    return len(data)

# 🎯 **Function to Perform Sentiment Analysis**
def analyze_sentiment():
    """Perform sentiment analysis on all stored posts and comments."""
    vader = SentimentIntensityAnalyzer()
    cursor.execute("SELECT id, content FROM reddit_data WHERE sentiment IS NULL")
    rows = cursor.fetchall()

    for row_id, text in rows:
        sentiment_score = vader.polarity_scores(text)["compound"] if len(text.split()) < 5 else TextBlob(text).sentiment.polarity
        sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
        cursor.execute("UPDATE reddit_data SET sentiment = ? WHERE id = ?", (sentiment, row_id))

    conn.commit()

# 🎯 **Function to Plot Sentiment Over Time**
def plot_sentiment_over_time():
    """Visualizes how the sentiment of user comments changed over time."""
    df = pd.read_sql("SELECT created_at, sentiment FROM reddit_data WHERE type='Comment'", conn)
    
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["Month"] = df["created_at"].dt.to_period("M")

    sentiment_counts = df.groupby(["Month", "sentiment"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    sentiment_counts.plot(kind="line", ax=ax, marker="o")

    ax.set_title("Sentiment of User Comments Over Time")
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Number of Comments")
    ax.legend(title="Sentiment")
    ax.grid(True)

    st.pyplot(fig)

# 🎯 **Function to Generate Word Cloud**
def generate_wordcloud(start_date, end_date):
    """Generates a word cloud for comments within a selected date range."""
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df_comments = pd.read_sql("SELECT content, created_at FROM reddit_data WHERE type='Comment'", conn)
    df_comments["created_at"] = pd.to_datetime(df_comments["created_at"])
    df_filtered = df_comments[(df_comments["created_at"] >= start_date) & (df_comments["created_at"] <= end_date)]

    if df_filtered.empty:
        st.write("No comments found for the selected time range.")
        return

    text = " ".join(df_filtered["content"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# 🎯 **Function for AI-Powered Queries**
def query_reddit_data(natural_language_query):
    """Allows users to query Reddit data using natural language."""
    engine = create_engine("sqlite:///reddit_analysis.db")
    db = SQLDatabase(engine)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

    response = db_chain.run(natural_language_query)
    return response

# 🎯 **Streamlit Web App**
st.title("📊 Reddit User Analysis Tool for 'opinionsareus'")

if st.button("Scrape & Analyze Data for 'opinionsareus'"):
    st.write("Fetching data...")
    num_entries = store_reddit_data(years=2)
    analyze_sentiment()
    st.success(f"✅ Scraped & Analyzed {num_entries} posts and comments!")

# 🎯 **Sentiment Over Time**
st.subheader("📉 Sentiment Over Time (Last 2 Years)")
plot_sentiment_over_time()

# 🎯 **Word Cloud with Date Selection**
st.subheader("☁️ Most Frequent Words in User's Comments (Select Time Period)")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.today().date() - timedelta(days=365*2))  
with col2:
    end_date = st.date_input("End Date", datetime.today().date())

if st.button("Generate Word Cloud"):
    generate_wordcloud(start_date, end_date)

# 🎯 **AI-Powered Querying**
st.header("🧠 AI-Powered Queries")
query = st.text_input("Ask a question about the user's Reddit activity:", "Show me all negative comments.")
if st.button("Run Query"):
    response = query_reddit_data(query)
    st.write("🧠 AI Response:", response)
