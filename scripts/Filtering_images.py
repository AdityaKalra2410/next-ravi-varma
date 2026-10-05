import os
import shutil
import json
import pandas as pd

# 1. Paths configuration
data_dir = "."  # Base directory
output_dir = "./processed_dataset"

# Search for captions_report.csv in current directory or ./data
csv_path = os.path.join(data_dir, "captions_report.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(data_dir, "data", "captions_report.csv")

# Search for images directory
images_dir = os.path.join(data_dir, "images")
if not os.path.exists(images_dir):
    images_dir = os.path.join(data_dir, "data", "images")

# Clean and recreate output directory
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# 2. Read captions_report.csv
df = pd.read_csv(csv_path)

# Normalize column names to lowercase
df.columns = [c.lower() for c in df.columns]

# Filter strictly for 'train' split (52 images)
train_df = df[df['split'].astype(str).str.lower() == 'train']

metadata_list = []
valid_count = 0
skipped_count = 0

# 3. Process train split images and captions
for _, row in train_df.iterrows():
    img_name = str(row['filename']).strip()
    caption_text = str(row['caption']).strip()
    
    src_img_path = os.path.join(images_dir, img_name)
    
    if os.path.exists(src_img_path) and caption_text:
        dst_img_path = os.path.join(output_dir, img_name)
        shutil.copy(src_img_path, dst_img_path)
        
        metadata_list.append({
            "file_name": img_name,
            "text": caption_text
        })
        valid_count += 1
    else:
        print(f"Warning: Image file not found: {src_img_path}")
        skipped_count += 1

# 4. Write metadata.jsonl inside output_dir
with open(os.path.join(output_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
    for entry in metadata_list:
        f.write(json.dumps(entry) + "\n")

print("--- Dataset Preparation Complete ---")
print(f"Total rows in CSV: {len(df)}")
print(f"Train split images processed: {valid_count}")
print(f"Test split images skipped: {len(df) - len(train_df)}")
if skipped_count > 0:
    print(f"Missing train images skipped: {skipped_count}")
print(f"Processed dataset ready at: {output_dir}")