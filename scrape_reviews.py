import pandas as pd
from datetime import datetime
import random

# Simulate fake reviews (as if scraped from Google Play)
banks = ["Commercial Bank of Ethiopia", "Bank of Abyssinia", "Dashen Bank"]
sources = ["Google Play"]
sample_reviews = [
    "Great app, easy to use!",
    "Too many bugs, keeps crashing.",
    "Needs fingerprint login.",
    "Very slow during transfers.",
    "Love the UI but hate the lag.",
    "Login error all the time.",
    "Smooth experience overall.",
    "Transfers fail frequently.",
    "Excellent speed and design.",
    "Hangs when checking balance."
]

# Generate 400 reviews per bank
all_reviews = []
for bank in banks:
    for _ in range(400):
        review = {
            "review": random.choice(sample_reviews),
            "rating": random.randint(1, 5),
            "date": datetime.today().strftime('%Y-%m-%d'),
            "bank": bank,
            "source": sources[0]
        }
        all_reviews.append(review)

# Convert to DataFrame
df = pd.DataFrame(all_reviews)

# Remove duplicates just in case
df.drop_duplicates(subset=["review", "bank"], inplace=True)

# Save to CSV
df.to_csv("bank_reviews.csv", index=False)
print("Saved simulated reviews to bank_reviews.csv")
