import pandas as pd
import re
import glob
from pathlib import Path

# Amharic text cleaner
def clean_amharic(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove links
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # remove mentions
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'[^\u1200-\u137F\s]+', '', text)  # keep only Amharic + space
    text = re.sub(r'\s+', ' ', text)  # normalize spaces
    return text.strip()

# Merge all CSVs
csv_files = glob.glob("*_messages.csv")
all_messages = pd.concat([pd.read_csv(file) for file in csv_files])

# Drop empty rows
all_messages = all_messages.dropna(subset=['message'])

# Clean messages
all_messages["cleaned_message"] = all_messages["message"].apply(clean_amharic)

# Drop duplicates
all_messages = all_messages.drop_duplicates(subset=["cleaned_message"])

# Save to new file
output_path = Path("cleaned_telegram_messages.csv")
all_messages.to_csv(output_path, index=False)

print(f"✅ Saved cleaned messages to {output_path} with {len(all_messages)} rows.")
