import pandas as pd

df = pd.read_csv("cleaned_telegram_messages.csv")

# Show the first 5 messages
print(df.head(5))
