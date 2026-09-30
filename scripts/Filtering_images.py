import os
import shutil
import pandas as pd
import json

# 1. Paths
data_dir = "."  # Current directory (/AI Course/next-ravi-varma/data)
output_dir = "./processed_dataset"
images_dir = os.path.join(data_dir, "images")
captions_dir = os.path.join(data_dir, "captions")

os.makedirs(output_dir, exist_ok=True)

metadata_list = []
valid_count = 0
skipped_count = 0

# Trigger token to guarantee style alignment
TRIGGER_TOKEN = "rrv_style"

# 2. Match images with captions
for img_name in os.listdir(images_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        continue
    
    base_name = os.path.splitext(img_name)[0]
    caption_file = os.path.join(captions_dir, f"{base_name}.txt")
    
    # Check if a non-empty caption text file exists
    if os.path.exists(caption_file):
        with open(caption_file, 'r', encoding='utf-8') as f:
            caption_text = f.read().strip()
        
        if caption_text:
            # Ensure trigger token is present in the caption
            if TRIGGER_TOKEN not in caption_text:
                caption_text = f"{caption_text}, {TRIGGER_TOKEN}"
                
            # Copy image to processed directory
            src_img_path = os.path.join(images_dir, img_name)
            dst_img_path = os.path.join(output_dir, img_name)
            shutil.copy(src_img_path, dst_img_path)
            
            # Record entry for metadata.jsonl
            metadata_list.append({
                "file_name": img_name,
                "text": caption_text
            })
            valid_count += 1
            continue
            
    skipped_count += 1

# 3. Write metadata.jsonl inside the processed dataset folder
with open(os.path.join(output_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
    for entry in metadata_list:
        f.write(json.dumps(entry) + "\n")

print(f"--- Dataset Preparation Complete ---")
print(f"Valid image-caption pairs copied: {valid_count}")
print(f"Uncaptioned images skipped: {skipped_count}")
print(f"Processed dataset ready at: {output_dir}")