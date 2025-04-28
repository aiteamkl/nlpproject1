# Community Pulse: Automated Sentiment Analysis from Reddit for East Bay Civic Insights
# Created based on feasibility study by Hung Lu & Jesse Katz
# CIS 96L NLP - Professors Sanjay Dorairaj & Tuan Nguyen


import os
from dotenv import load_dotenv
import praw
import pandas as pd
import datetime
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import spacy
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import pymongo
from pymongo import MongoClient
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

load_dotenv()

# Initialize NLTK components
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Load spaCy for NER
nlp = spacy.load('en_core_web_sm')


# Initialize sentiment analysis model
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


# Reddit API Configuration
def configure_reddit_api():
    """Configure and return Reddit API client"""
    reddit = praw.Reddit(
        client_id=os.environ.get('REDDIT_CLIENT_ID', 'Fj0Sk2BLVJ0PCBA_r41zEg'),
        client_secret=os.environ.get('REDDIT_CLIENT_SECRET', 'arhY7vtO1j-XiMsyyiwl4kKWI_PLeQ'),
        user_agent="Community Pulse by /u/Monk481"
    )
    return reddit


# Database Configuration
def configure_database():
    """Configure and return MongoDB client"""
    try:
        client = MongoClient(os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/'))
        db = client['community_pulse']
        return db
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None


# Text Preprocessing
def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
   
    # Convert to lowercase and remove URLs
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
   
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
   
    # Tokenize
    tokens = word_tokenize(text)
   
    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
   
    return ' '.join(cleaned_tokens)


# Data Collection
def scrape_reddit_data(subreddits=['oakland', 'eastbay', 'BayArea'], time_filter='week', limit=100):
    """Scrape data from specified subreddits"""
    reddit = configure_reddit_api()
    all_posts = []
   
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
           
            # Get posts
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'created_utc': datetime.datetime.fromtimestamp(post.created_utc),
                    'score': post.score,
                    'comments': post.num_comments,
                    'subreddit': subreddit_name,
                    'url': post.url,
                    'author': str(post.author),
                    'processed_text': preprocess_text(f"{post.title} {post.selftext}"),
                }
               
                # Get top comments
                post.comments.replace_more(limit=0)
                post_data['top_comments'] = []
               
                for comment in post.comments.list()[:10]:  # Get top 10 comments
                    if hasattr(comment, 'body'):
                        comment_data = {
                            'id': comment.id,
                            'text': comment.body,
                            'created_utc': datetime.datetime.fromtimestamp(comment.created_utc),
                            'score': comment.score,
                            'processed_text': preprocess_text(comment.body),
                        }
                        post_data['top_comments'].append(comment_data)
               
                all_posts.append(post_data)
               
        except Exception as e:
            st.error(f"Error scraping r/{subreddit_name}: {e}")
   
    return all_posts


# Sentiment Analysis
def analyze_sentiment(text):
    """Analyze sentiment of text using transformer model"""
    if not text or text.strip() == "":
        return {"label": "neutral", "score": 0.5}
   
    try:
        result = sentiment_model(text[:512])  # Truncate to avoid token limit
        return result[0]
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return {"label": "neutral", "score": 0.5}


# Named Entity Recognition
def extract_entities(text):
    """Extract entities like locations, organizations from text"""
    if not text or text.strip() == "":
        return []
   
    doc = nlp(text)
    entities = {}
   
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
   
    return entities


# Topic Modeling
def extract_topics(texts, n_topics=10):
    """Extract main topics from a collection of texts"""
    if not texts or len(texts) < 5:  # Need sufficient data for topic modeling
        return [{"id": 0, "name": "Insufficient data", "keywords": []}]
   
    try:
        # Initialize BERTopic
        topic_model = BERTopic(nr_topics=n_topics)
       
        # Fit the model (might take time for large datasets)
        topics, _ = topic_model.fit_transform(texts)
       
        # Get topic information
        topic_info = topic_model.get_topic_info()
        topic_keywords = {}
       
        for topic in topic_info['Topic']:
            if topic != -1:  # Skip outlier topic
                words = [word for word, _ in topic_model.get_topic(topic)]
                topic_keywords[topic] = words[:5]  # Get top 5 keywords
       
        # Format for return
        formatted_topics = [
            {"id": topic, "name": f"Topic {topic}", "keywords": keywords}
            for topic, keywords in topic_keywords.items()
        ]
       
        return formatted_topics
    except Exception as e:
        print(f"Error in topic modeling: {e}")
        return [{"id": 0, "name": "Error in topic modeling", "keywords": []}]


# Keyword Extraction
def extract_keywords(text, n=10):
    """Extract key terms using TF-IDF"""
    if not text or text.strip() == "":
        return []
   
    try:
        # Use TF-IDF for keyword extraction
        vectorizer = TfidfVectorizer(max_features=n)
        tfidf_matrix = vectorizer.fit_transform([text])
       
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
       
        # Get scores
        scores = tfidf_matrix.toarray()[0]
       
        # Create tuples of terms and scores
        tuples = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
       
        # Sort by score
        tuples.sort(key=lambda x: x[1], reverse=True)
       
        # Return top n keywords
        return [word for word, score in tuples[:n]]
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []


# Store data in MongoDB
def store_data(data, db):
    """Store scraped and processed data in MongoDB"""
    if not db:
        return False
   
    try:
        posts_collection = db['posts']
       
        # Add sentiment, entities, and keywords to posts
        for post in data:
            # Analyze sentiment for post
            sentiment = analyze_sentiment(post['processed_text'])
            post['sentiment'] = sentiment
           
            # Extract entities
            post['entities'] = extract_entities(post['processed_text'])
           
            # Extract keywords
            post['keywords'] = extract_keywords(post['processed_text'])
           
            # Process comments
            for comment in post['top_comments']:
                comment_sentiment = analyze_sentiment(comment['processed_text'])
                comment['sentiment'] = comment_sentiment
           
            # Store timestamp for tracking
            post['processed_at'] = datetime.datetime.now()
           
            # Store in MongoDB (update if exists, insert if new)
            posts_collection.update_one(
                {'id': post['id']},
                {'$set': post},
                upsert=True
            )
       
        return True
    except Exception as e:
        st.error(f"Error storing data: {e}")
        return False


def load_data(db, time_filter='week'):
    """Load data from MongoDB with optional time filter"""
    if not db:
        return []
   
    try:
        posts_collection = db['posts']
       
        # Calculate time threshold
        if time_filter == 'day':
            threshold = datetime.datetime.now() - datetime.timedelta(days=1)
        elif time_filter == 'week':
            threshold = datetime.datetime.now() - datetime.timedelta(weeks=1)
        elif time_filter == 'month':
            threshold = datetime.datetime.now() - datetime.timedelta(days=30)
        else:  # all time
            threshold = datetime.datetime(2000, 1, 1)
       
        # Query database
        cursor = posts_collection.find({'created_utc': {'$gte': threshold}})
       
        # Convert to list and handle MongoDB ObjectId
        posts = list(cursor)
        for post in posts:
            post['_id'] = str(post['_id'])
       
        return posts
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return []


# Generate insights
def generate_insights(posts):
    """Generate insights from posts data"""
    if not posts:
        return {
            "sentiment_summary": {"positive": 0, "negative": 0, "neutral": 0},
            "top_topics": [],
            "trending_keywords": [],
            "location_mentions": {},
        }
   
    insights = {}
   
    # Sentiment analysis summary
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for post in posts:
        sentiment = post.get('sentiment', {}).get('label', 'neutral')
        if sentiment == "POSITIVE":
            sentiment_counts["positive"] += 1
        elif sentiment == "NEGATIVE":
            sentiment_counts["negative"] += 1
        else:
            sentiment_counts["neutral"] += 1
   
    insights["sentiment_summary"] = sentiment_counts
   
    # Extract all preprocessed texts for topic modeling
    all_texts = [post['processed_text'] for post in posts if post.get('processed_text')]
   
    # Get topics
    if len(all_texts) > 5:  # Need at least 5 documents for meaningful topics
        insights["top_topics"] = extract_topics(all_texts)
    else:
        insights["top_topics"] = []
   
    # Get trending keywords
    all_keywords = []
    for post in posts:
        keywords = post.get('keywords', [])
        all_keywords.extend(keywords)
   
    keyword_counter = Counter(all_keywords)
    insights["trending_keywords"] = keyword_counter.most_common(15)
   
    # Get location mentions
    location_mentions = {}
    for post in posts:
        entities = post.get('entities', {})
        locations = entities.get('GPE', []) + entities.get('LOC', [])
       
        for location in locations:
            if location.lower() not in location_mentions:
                location_mentions[location.lower()] = 0
            location_mentions[location.lower()] += 1
   
    insights["location_mentions"] = location_mentions
   
    return insights


# Visualization Functions
def create_sentiment_chart(data):
    """Create sentiment distribution chart"""
    sentiment_data = data["sentiment_summary"]
   
    fig = px.pie(
        values=[sentiment_data["positive"], sentiment_data["negative"], sentiment_data["neutral"]],
        names=["Positive", "Negative", "Neutral"],
        title="Sentiment Distribution",
        color_discrete_sequence=["#2E8B57", "#CD5C5C", "#6495ED"]
    )
   
    return fig


def create_wordcloud(keywords):
    """Create wordcloud from keywords"""
    if not keywords:
        return None
   
    # Create a dictionary of words and their frequencies
    word_freq = {}
    for word, count in keywords:
        word_freq[word] = count
   
    # Generate wordcloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate_from_frequencies(word_freq)
   
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
   
    return fig


def create_topic_chart(topics):
    """Create horizontal bar chart for topic distribution"""
    if not topics:
        return None
   
    topic_names = [f"Topic {topic['id']}: {', '.join(topic['keywords'][:3])}" for topic in topics]
    topic_sizes = [1] * len(topics)  # Placeholder for actual topic sizes
   
    fig = px.bar(
        x=topic_sizes,
        y=topic_names,
        orientation='h',
        title="Topic Distribution",
        labels={"x": "Count", "y": "Topic"}
    )
   
    return fig


def create_location_map(location_data):
    """Create a map with location mentions"""
    if not location_data:
        return None
   
    # East Bay area map centered roughly on Oakland
    m = folium.Map(location=[37.8044, -122.2711], zoom_start=11)
   
    # Placeholder for demo - in a real app, you'd geocode these locations
    # This is a mockup for visualization purposes
    sample_locations = {
        "oakland": [37.8044, -122.2711],
        "berkeley": [37.8715, -122.2730],
        "alameda": [37.7652, -122.2416],
        "richmond": [37.9358, -122.3478],
        "hayward": [37.6688, -122.0810],
        "san leandro": [37.7249, -122.1561],
        "emeryville": [37.8312, -122.2852],
        "el cerrito": [37.9156, -122.3108],
        "albany": [37.8868, -122.2977],
        "castro valley": [37.6941, -122.0858],
        "san lorenzo": [37.6810, -122.1244],
        "concord": [37.9779, -122.0301],
        "walnut creek": [37.9101, -122.0652],
        "fremont": [37.5485, -121.9886],
        "union city": [37.5936, -122.0438],
        "livermore": [37.6819, -121.7680],
        "pleasanton": [37.6624, -121.8747],
        "dublin": [37.7021, -121.9358],
        "piedmont": [37.8243, -122.2316],
        "martinez": [38.0194, -122.1341],
        "pittsburg": [38.0279, -121.8850],
        "antioch": [38.0049, -121.8058],
        "brentwood": [37.9319, -121.6958]
    }
   
    # Create heat map data
    heat_data = []
    for location, count in location_data.items():
        location_lower = location.lower()
        if location_lower in sample_locations:
            coords = sample_locations[location_lower]
            # Add a point for each mention
            for _ in range(count):
                heat_data.append(coords)
   
    # Add heat map layer
    if heat_data:
        HeatMap(heat_data).add_to(m)
   
    return m


# Streamlit Dashboard
def build_dashboard():
    """Build and display the Streamlit dashboard"""
    # Set page config
    st.set_page_config(
        page_title="Community Pulse",
        page_icon="📊",
        layout="wide"
    )
   
    # Header
    st.title("Community Pulse: East Bay Civic Insights")
    st.subheader("Automated Sentiment Analysis from Reddit")
   
    # Sidebar for filters
    st.sidebar.header("Filters")
   
    # Subreddit selection
    subreddits = st.sidebar.multiselect(
        "Select Subreddits",
        ["oakland", "eastbay", "BayArea", "berkeley", "SanFrancisco"],
        default=["oakland", "eastbay", "BayArea"]
    )
   
    # Time filter
    time_filter = st.sidebar.selectbox(
        "Time Period",
        ["day", "week", "month", "all"],
        index=1
    )
   
    # Action buttons
    if st.sidebar.button("Refresh Data"):
        with st.spinner("Scraping new data from Reddit..."):
            # Configure database
            db = configure_database()
           
            # Scrape new data
            posts = scrape_reddit_data(subreddits, time_filter)
           
            if posts:
                # Store data
                success = store_data(posts, db)
                if success:
                    st.sidebar.success(f"Successfully scraped {len(posts)} posts!")
                else:
                    st.sidebar.error("Failed to store data.")
            else:
                st.sidebar.error("No data scraped.")
   
    # Load data for dashboard
    db = configure_database()
    posts = load_data(db, time_filter)
   
    # Main dashboard
    if not posts:
        st.info("No data available. Please scrape data using the 'Refresh Data' button.")
    else:
        # Generate insights
        insights = generate_insights(posts)
       
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Posts", len(posts))
        with col2:
            avg_sentiment = (
                insights["sentiment_summary"]["positive"] -
                insights["sentiment_summary"]["negative"]
            ) / len(posts) if len(posts) > 0 else 0
            st.metric("Sentiment Score", f"{avg_sentiment:.2f}")
        with col3:
            total_comments = sum(post.get('comments', 0) for post in posts)
            st.metric("Total Comments", total_comments)
       
        # Sentiment chart
        st.subheader("Sentiment Analysis")
        sentiment_chart = create_sentiment_chart(insights)
        st.plotly_chart(sentiment_chart, use_container_width=True)
       
        # Create two columns
        col1, col2 = st.columns(2)
       
        with col1:
            # Wordcloud
            st.subheader("Trending Keywords")
            wordcloud_fig = create_wordcloud(insights["trending_keywords"])
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            else:
                st.info("Not enough data for wordcloud.")
       
        with col2:
            # Topic distribution
            st.subheader("Topic Distribution")
            topic_chart = create_topic_chart(insights["top_topics"])
            if topic_chart:
                st.plotly_chart(topic_chart, use_container_width=True)
            else:
                st.info("Not enough data for topic modeling.")
       
        # Map visualization
        st.subheader("Geographic Insights")
        location_map = create_location_map(insights["location_mentions"])
        if location_map:
            folium_static(location_map)
        else:
            st.info("No location data available.")
       
        # Recent posts
        st.subheader("Recent Posts")
        for i, post in enumerate(posts[:5]):  # Show only 5 most recent posts
            with st.expander(f"{post['title']} (from r/{post['subreddit']})"):
                st.write(f"**Created:** {post['created_utc']}")
                st.write(f"**Score:** {post['score']}")
                st.write(f"**Comments:** {post['comments']}")
               
                # Show sentiment with color
                sentiment = post.get('sentiment', {}).get('label', 'NEUTRAL')
                score = post.get('sentiment', {}).get('score', 0.5)
               
                if sentiment == "POSITIVE":
                    st.markdown(f"**Sentiment:** <span style='color:green'>Positive ({score:.2f})</span>", unsafe_allow_html=True)
                elif sentiment == "NEGATIVE":
                    st.markdown(f"**Sentiment:** <span style='color:red'>Negative ({score:.2f})</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Sentiment:** <span style='color:blue'>Neutral ({score:.2f})</span>", unsafe_allow_html=True)
               
                # Show post content
                if post.get('text'):
                    st.write("**Content:**")
                    st.write(post['text'][:500] + "..." if len(post['text']) > 500 else post['text'])
               
                # Show keywords
                if post.get('keywords'):
                    st.write("**Keywords:**", ", ".join(post.get('keywords', [])[:10]))


# Run the app
if __name__ == "__main__":
    build_dashboard()

