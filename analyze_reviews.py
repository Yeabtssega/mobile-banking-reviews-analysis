import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon (only needed once)
nltk.download('vader_lexicon')

# Load the CSV with reviews
df = pd.read_csv("bank_reviews.csv")

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Calculate sentiment scores and classify sentiment
def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment"] = df["review"].apply(get_sentiment)

# Define some keywords/themes to look for
keywords = {
    "bugs": ["bug", "crash", "error", "fail", "hang"],
    "ui": ["ui", "interface", "design", "look", "layout"],
    "speed": ["slow", "speed", "lag", "loading"],
    "login": ["login", "sign in", "authentication", "password"],
    "transfer": ["transfer", "send money", "payment", "transaction"],
    "fingerprint": ["fingerprint", "biometric", "touch id"],
}

# Function to find matching themes in a review
def find_themes(text):
    found = []
    text_lower = text.lower()
    for theme, kws in keywords.items():
        for kw in kws:
            if kw in text_lower:
                found.append(theme)
                break
    return found if found else ["other"]

df["themes"] = df["review"].apply(find_themes)

# Explode the themes list to one theme per row for easier aggregation
df_exp = df.explode("themes")

# Save the enriched dataset
df.to_csv("bank_reviews_with_sentiment.csv", index=False)

# --- VISUALIZATION ---

# 1) Sentiment distribution per bank
sentiment_counts = df.groupby(["bank", "sentiment"]).size().unstack(fill_value=0)
sentiment_counts.plot(kind="bar", stacked=True, figsize=(8,6), colormap="Paired")
plt.title("Sentiment Distribution per Bank")
plt.xlabel("Bank")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.close()

# 2) Top themes per bank
theme_counts = df_exp.groupby(["bank", "themes"]).size().unstack(fill_value=0)
theme_counts.plot(kind="bar", stacked=True, figsize=(10,6), colormap="Set3")
plt.title("Theme Frequency per Bank")
plt.xlabel("Bank")
plt.ylabel("Number of Mentions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("theme_frequency.png")
plt.close()

print("Analysis complete. Saved 'bank_reviews_with_sentiment.csv' and visualizations.")
