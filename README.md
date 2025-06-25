# FinTech_Vendor_Scorecard


# 🧠 EthioMart FinTech Vendor Scorecard

A data-driven analytics engine that evaluates small vendors on Telegram based on engagement metrics and business activity to generate a **lending score** for micro-lending decisions.


## 📌 Project Overview

This project scrapes public Telegram posts from EthioMart vendor channels, extracts key business entities (like product names and prices) using NLP, and generates a **Vendor Lending Scorecard** — helping FinTech lenders identify promising small businesses for micro-loans.


## 🧩 Tasks Covered

| Task | Description |
|------|-------------|
| **1** | Scraped public Telegram channel messages |
| **2** | Preprocessed text for structured analysis |
| **3** | Built and trained a custom NER model to extract entities |
| **4** | Integrated scraped posts with extracted entities |
| **5** | Profiled each vendor's business behavior |
| **6** | Generated a vendor lending score based on engagement and consistency |


## 📁 Folder Structure

```
FinTech_Vendor_Scorecard/
│
├── data/                   # Input and output data files
│   ├── posts.csv           # Raw Telegram posts
│   ├── cleaned_posts.csv   # Cleaned post texts
│   ├── ner_entities.csv    # Extracted NER entities
│   └── vendor_lending_scorecard.csv  # Final vendor scores
│
├── src/                    # Source code scripts
│   ├── scraping/           # Telegram scraping tools
│   │   └── telegram_scraper.py
│   ├── preprocessing/      # Text cleaning utilities
│   │   └── text_cleaner.py
│   ├── ner/                # NER model training & extraction
│   │   └── train_ner_model.py
│   └── scorecard/          # Analytics engine & scoring logic
│       └── analytics_engine.py
│
├── README.md               # You are here!
└── requirements.txt        # Python dependencies
```


## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/FinTech_Vendor_Scorecard.git
cd FinTech_Vendor_Scorecard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: Some components like `spaCy` may require additional language models:
```bash
python -m spacy download en_core_web_sm
```


## 🚀 How to Run

### Step 1: Scrape Telegram Posts
```bash
python src/scraping/telegram_scraper.py
```

### Step 2: Clean the Posts
```bash
python src/preprocessing/text_cleaner.py
```

### Step 3: Train or Apply NER Model
```bash
python src/ner/train_ner_model.py
```

### Step 4: Generate Vendor Lending Scores
```bash
python src/scorecard/analytics_engine.py
```

✅ Output will be saved as `data/vendor_lending_scorecard.csv`.


## 📊 Sample Output (vendor_lending_scorecard.csv)

| channel_id     | posting_frequency | avg_views | top_post_views | top_product     | top_price | avg_price | lending_score |
|----------------|------------------|-----------|----------------|------------------|-----------|-----------|---------------|
| ethiomart_001  | 4.3              | 820       | 1650           | Women’s Dress   | 950       | 780       | 935           |
| ethiomart_002  | 2.1              | 310       | 620            | Men’s Shoes     | 600       | 550       | 465           |



## 🛠️ Customization Options

- Modify the **lending score formula** in `analytics_engine.py`
- Add new entity types to the **NER model**
- Improve text cleaning rules in `text_cleaner.py`
- Add **sentiment analysis**, **product categorization**, or **growth trend tracking**


