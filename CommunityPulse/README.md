# Community Pulse 🌐📊

**Automated Sentiment Analysis Dashboard for East Bay Civic Insights**

---

## Overview

Community Pulse scrapes Reddit discussions from East Bay community subreddits (like r/oakland, r/eastbay)  
to analyze public sentiment, identify emerging topics, and visualize civic trends in real time.

It provides an interactive Streamlit dashboard with:

- 📅 Sentiment trends over time
- 💬 Word clouds of trending keywords
- 🔥 Top emerging discussion topics
- 🗺️ Geographic heatmaps for city mentions
- 📰 Expandable views of recent posts

---

## Features

✅ Real-time Reddit scraping (r/oakland, r/eastbay, r/BayArea, and more)  
✅ Sentiment classification using a fine-tuned Transformer model (DistilBERT)  
✅ Named Entity Recognition (NER) to extract locations, organizations  
✅ Topic modeling with BERTopic  
✅ Keyword extraction with TF-IDF  
✅ Interactive Streamlit dashboard (charts, maps, and data exploration)  
✅ MongoDB backend for persistent storage of posts and analysis  
✅ Ready for cloud deployment or local server hosting

---

## How to Run Locally

1. **Clone this repository:**

```bash
git clone https://github.com/aiteamkl/nlpproject1.git

Navigate to the project folder:

bash
Copy
Edit
cd nlpproject1/CommunityPulse
Create and activate a virtual environment:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate      # (Windows)
source venv/bin/activate   # (Mac/Linux)
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Set your environment variables (MongoDB URI, etc.)

Create a .env file with:

ini
Copy
Edit
MONGODB_URI=mongodb://your-server-ip:27017/
Run the Streamlit dashboard:

bash
Copy
Edit
python -m streamlit run community_pulse_app.py
✅ Dashboard will open automatically at http://localhost:8501/.

Tech Stack
Python (3.10+)

Streamlit (Dashboard UI)

MongoDB (Database backend)

Hugging Face Transformers (Sentiment analysis)

BERTopic (Topic modeling)

spaCy (NER)

NLTK (Text preprocessing)

Plotly, WordCloud, Folium (Visualizations)

Contributors
Jesse Katz

Hung Lu

Future Enhancements
Email/SMS alerts for trending issues

Scheduled auto-scraping via CRON jobs

Deeper sarcasm/irony detection in sentiment models

Geo-coded location extraction for precise mapping

