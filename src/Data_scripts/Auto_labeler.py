import pandas as pd

# Load your datasets
labeling_template = pd.read_csv("Data/sample_label.csv")
unlabeled_data = pd.read_csv("Data/amharic_ner_labeling_sheet2.csv")

# Rename columns if needed
if "token" in unlabeled_data.columns:
    unlabeled_data.rename(columns={"token": "token_to_label"}, inplace=True)

# Ensure consistent message grouping
def group_tokens_by_message(df):
    return df.groupby("message_id")["token_to_label"].apply(list).reset_index(name="tokens")

# Group tokens by message
grouped_unlabeled = group_tokens_by_message(unlabeled_data)

# Create a list of labeled rows
labeled_rows = []

for _, row in labeling_template.iterrows():
    message_id = row["message_id"]
    token = row["token"]
    label = row["label"]

    labeled_rows.append({
        "message_id": message_id,
        "token": token,
        "label": label
    })

# Save as new CSV with correct format
labeled_df = pd.DataFrame(labeled_rows)
labeled_df.to_csv("amharic_ner_labeling_sheet_aligned2.csv", index=False)

print("✅ Labeled file saved as 'amharic_ner_labeling_sheet_aligned.csv'")
