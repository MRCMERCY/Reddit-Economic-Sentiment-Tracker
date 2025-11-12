# economic_sentiment_full_pro_live.py
# Advanced Economic Sentiment Tracker – Multi-Domain & Multi-Source
# By Marek
# Features: FinBERT, multi-domain sentiment, Reddit + news API, trend + event detection, CSV logging

import os
import time
import csv
import praw
import requests
import torch
import pandas as pd
from datetime import datetime
from collections import Counter, deque
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- CONFIG ----------------
load_dotenv()
CLIENT_ID     = os.getenv("CLIENT_ID")       # Reddit
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT    = os.getenv("USER_AGENT", "econ-sentiment-pro")

NEWS_API_KEY  = os.getenv("NEWS_API_KEY")    # News API

FETCH_INTERVAL      = 30      # seconds
POST_LIMIT          = 50
RUNNING_AVG_LENGTH  = 10
SUBREDDITS          = "worldnews+economics+politics+news"
LOG_CSV_FILE        = "econ_sentiment_full_log.csv"

# ---------------- DOMAIN KEYWORDS ----------------
KEYWORDS = {
    "Inflation/Prices": ["inflation","price","rent","cost-of-living","gas","mortgage","energy","housing","tax","interest rate"],
    "Employment": ["unemployment","jobless","hiring","layoffs","job creation","recruitment","firing"],
    "Housing": ["mortgage","housing","rent","real estate","affordability","property"],
    "Policy": ["interest rate","fiscal","monetary","central bank","regulation","policy","stimulus","tax"],
    "GDP/Trade": ["gdp","growth","exports","imports","trade","surplus","deficit","economy"]
}

# ---------------- INIT ----------------
reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)

MODEL_NAME = "ProsusAI/finbert"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model      = model.to(device)

trend_history = {domain: deque() for domain in KEYWORDS.keys()}

# ---------------- FUNCTIONS ----------------
def fetch_reddit_posts(country):
    query = f"{country} economy OR inflation OR prices OR jobs OR GDP OR policy OR rent OR mortgage"
    posts = []
    for submission in reddit.subreddit(SUBREDDITS).search(query, limit=POST_LIMIT, sort="new"):
        posts.append({"title": submission.title, "body": submission.selftext})
    return posts

def fetch_news_posts(country):
    url = f"https://newsapi.org/v2/everything?q={country}+economy&apiKey={NEWS_API_KEY}&language=en&pageSize={POST_LIMIT}"
    try:
        r = requests.get(url)
        data = r.json()
        articles = data.get("articles", [])
        posts = [{"title": a["title"], "body": a["description"] or ""} for a in articles]
        return posts
    except Exception as e:
        print("News API fetch error:", e)
        return []

def finbert_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    return scores[2] - scores[0]  # positive - negative

def domain_weighted_score(text):
    base_score = finbert_sentiment(text)
    text_lower = text.lower()
    domain_scores = {}
    for domain, kws in KEYWORDS.items():
        score = 0.0
        for kw in kws:
            if kw in text_lower:
                score += 0.25 if domain in ["GDP/Trade","Employment"] else 0.3
        domain_scores[domain] = max(min(base_score + score, 1.0), -1.0)
    return domain_scores

def analyze_posts(posts):
    if not posts:
        return {domain: 0.0 for domain in KEYWORDS.keys()}, {}, {}
    domain_avg = {domain: 0.0 for domain in KEYWORDS.keys()}
    all_neg_counter = Counter()
    all_pos_counter = Counter()
    for p in posts:
        text = p["title"] + " " + p["body"]
        scores = domain_weighted_score(text)
        for domain in KEYWORDS.keys():
            domain_avg[domain] += scores[domain]
        # topic counting
        t = text.lower()
        for domain, kws in KEYWORDS.items():
            for kw in kws:
                if kw in t:
                    if scores[domain] < -0.2:
                        all_neg_counter[kw] +=1
                    elif scores[domain] > 0.2:
                        all_pos_counter[kw] +=1
    for domain in domain_avg.keys():
        domain_avg[domain] /= len(posts)
    top_neg = [k for k,_ in all_neg_counter.most_common(5)]
    top_pos = [k for k,_ in all_pos_counter.most_common(5)]
    return domain_avg, top_neg, top_pos

def update_trends(domain_avg):
    running_avg = {}
    for domain, score in domain_avg.items():
        trend_history[domain].append(score)
        if len(trend_history[domain]) > RUNNING_AVG_LENGTH:
            trend_history[domain].popleft()
        running_avg[domain] = sum(trend_history[domain])/len(trend_history[domain])
    return running_avg

def log_csv(timestamp, country, running_avg, top_neg, top_pos):
    header = ["timestamp","country"] + list(KEYWORDS.keys()) + ["top_neg","top_pos"]
    exists = os.path.isfile(LOG_CSV_FILE)
    with open(LOG_CSV_FILE,"a",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        row = [timestamp, country] + [f"{running_avg[d]:.4f}" for d in KEYWORDS.keys()] + [";".join(top_neg), ";".join(top_pos)]
        writer.writerow(row)

def display_dashboard(country, running_avg, top_neg, top_pos):
    print("\n"+"="*70)
    print(f"🌍 Country: {country}")
    for domain, score in running_avg.items():
        bar_len = 20
        filled = int((score + 1)/2 * bar_len)
        bar = "🟩"*filled + "⬜"*(bar_len-filled)
        label = ("😡 Strongly Negative" if score < -0.4 else
                 "😐 Neutral" if abs(score)<=0.2 else
                 "😊 Positive")
        print(f"{domain:15}: {score:+.2f} {label} {bar}")
    print("\nTop Negative Topics:", ", ".join(top_neg) if top_neg else "None")
    print("Top Positive Topics:", ", ".join(top_pos) if top_pos else "None")
    print(f"Last update: {datetime.now().strftime('%I:%M:%S %p')}")
    print("="*70)

# ---------------- MAIN LOOP ----------------
def main():
    print("🌍 Advanced Economic Sentiment Tracker – FULL PRO")
    country = input("Enter a country name: ").strip()
    print(f"➡️ Monitoring economic sentiment for '{country}'...\n")
    while True:
        print(f"\n🔄 Fetching latest posts about {country} economy...")
        reddit_posts = fetch_reddit_posts(country)
        news_posts   = fetch_news_posts(country)
        all_posts = reddit_posts + news_posts
        domain_avg, top_neg, top_pos = analyze_posts(all_posts)
        running_avg = update_trends(domain_avg)
        log_csv(datetime.now().isoformat(), country, running_avg, top_neg, top_pos)
        display_dashboard(country, running_avg, top_neg, top_pos)
        for _ in tqdm(range(FETCH_INTERVAL), desc="Waiting for next update", ncols=70):
            time.sleep(1)

if __name__=="__main__":
    main()
