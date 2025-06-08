# Mobile Banking Reviews Analysis

This project analyzes customer reviews from Ethiopian mobile banking apps—CBE, BOA, and Dashen Bank—using web scraping, sentiment analysis, and thematic analysis. The goal is to provide actionable insights to improve user experience, customer satisfaction, and feature development.

---

## Project Overview

- **Data Collection:** Scraped real user reviews from Google Play Store for the three banks' mobile apps.
- **Sentiment Analysis:** Used NLP techniques to classify reviews as positive, neutral, or negative.
- **Thematic Analysis:** Extracted key themes and topics from the reviews using TF-IDF and clustering.
- **Database Engineering:** Stored cleaned and processed review data in a PostgreSQL database.
- **Visualization:** Created plots for sentiment distribution and theme frequency to highlight trends.
- **Final Report:** Summarizes insights and recommendations for mobile banking product teams.

---

## Data Sources

- **Apps Scraped:**
  - CBE Mobile Banking (`com.combanketh.mobilebanking`)
  - BOA Mobile Banking (`com.boa.boaMobileBanking`)
  - Dashen Super App (`com.dashen.dashensuperapp`)
- **Review Data:** Collected via `scrape_reviews.py`

---

## Methodology

1. **Web Scraping:**  
   Automated review extraction from Google Play Store using Python.

2. **Preprocessing:**  
   Cleaned and filtered raw reviews for analysis.

3. **Sentiment Analysis:**  
   Used VADER and TextBlob to assign sentiment scores and labels.

4. **Thematic Analysis:**  
   Applied TF-IDF and clustering to identify recurring topics and customer concerns.

5. **Data Storage:**  
   Inserted cleaned reviews and metadata into PostgreSQL database using `insert_reviews.py`.

6. **Visualization:**  
   Generated sentiment distribution and theme frequency plots for reporting.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Yeabtssega/mobile-banking-reviews-analysis.git
   cd mobile-banking-reviews-analysis
Install dependencies:


pip install -r requirements.txt
Run the scraper:


python scrape_reviews.py
Perform sentiment and thematic analysis:


python task2_sentiment_thematic.py
Insert data into PostgreSQL:


python insert_reviews.py
Generate visualizations:


python task2_visualize_results.py
Results
Sentiment distribution across apps reveals the overall user satisfaction levels.

Thematic analysis identifies common pain points and requested features.

Visualizations are saved in the outputs/ directory.

Dependencies
Python 3.11+

pandas

numpy

matplotlib

scikit-learn

nltk

vaderSentiment

psycopg2-binary

See requirements.txt for the full list.

Future Work
Expand scraping to include iOS App Store reviews.

Use advanced transformers like DistilBERT for improved sentiment accuracy.

Build a dashboard for interactive visualization and monitoring.

Conduct temporal analysis to track sentiment trends over time.

