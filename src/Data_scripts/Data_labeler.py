#src/labeler.py

import pandas as pd
from tqdm import tqdm
import re

def clean_amharic_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs, emojis, mentions, normalize spaces
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)  # Remove emojis
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_amharic(text):
    if not isinstance(text, str):
        return []
    # Basic whitespace + punctuation-based tokenizer
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text, re.UNICODE)
    return tokens

# Load your data
df = pd.read_csv('D:/PYTHON PROJECTS/KIAM PROJECTS/Amharic-E-commerce-Data-Extractor/Data/telegram_data.csv')

# Clean and tokenize
df['cleaned_text'] = df['text'].apply(clean_amharic_text)
df = df[df['cleaned_text'].str.len() > 10]  # Filter out short texts
df = df.sample(n=50, random_state=42).reset_index(drop=True)  # Sample 50 posts

# Save sampled data
df.to_csv('Data/labeling_sample_backup.csv', index=False)
print("✅ Sampled 50 messages for labeling.")

# Prepare rows for labeling
labeled_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    tokens = tokenize_amharic(row['cleaned_text'])  # Tokenize cleaned text
    for token in tokens:
        labeled_rows.append({
            'message_id': idx,
            'token': token,
            'label': ''  # Empty label placeholder
        })

# Save labeling sheet
labeling_df = pd.DataFrame(labeled_rows)
labeling_df.to_csv('Data/amharic_ner_labeling_sheet.csv', index=False)
print("✅ Labeling sheet saved to amharic_ner_labeling_sheet.csv")


# Load your labeled CSV file
labeling_df = pd.read_csv('Data/amharic_ner_labeling_sheet.csv')

# Open output file for writing
with open('amharic_ner.conll', 'w', encoding='utf-8') as f:
    current_message = None
    for _, row in labeling_df.iterrows():
        if row['message_id'] != current_message:
            current_message = row['message_id']
            f.write('\n')  # New message, add blank line
        token = str(row['token']).strip()
        label = str(row['label']).strip()
        if token:
            f.write(f"{token}\t{label}\n")


