import pandas as pd

# Load cleaned Telegram messages
df = pd.read_csv("cleaned_telegram_messages.csv")

# Drop rows where the message is NaN
df = df.dropna(subset=['cleaned_message'])

# Reset index and take 10 samples
sample_df = df[['cleaned_message']].drop_duplicates().reset_index(drop=True).head(10)

# Save to a new text file for annotation
with open("messages_for_annotation.txt", "w", encoding="utf-8") as f:
    for idx, row in sample_df.iterrows():
        f.write(f"# Message {idx + 1}\n")
        f.write(row['cleaned_message'] + "\n\n")

print("✅ Saved 10 messages to messages_for_annotation.txt")
