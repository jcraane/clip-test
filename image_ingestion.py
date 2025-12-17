import os
import torch
import clip
from PIL import Image
from pathlib import Path
import psycopg2
from tqdm import tqdm

DB_PARAMS = {
    "dbname": "embeddings",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5441"
}

# 1. Configuration
IMAGE_DIR = Path("~/Downloads/img").expanduser()

# 2. Load CLIP
device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_embedding(image_path):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
            # Normalize for cosine similarity
            embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return None

# 3. Connect to Postgres
conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()

# 4. Process Loop
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Recurse into subfolders (rglob is recursive)
files = [
    p for p in IMAGE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in valid_extensions
]

print(f"Found {len(files)} images. Starting indexing...")

for file_path in tqdm(files):
    vector = get_embedding(file_path)
    if vector:
        cur.execute(
            "INSERT INTO image_library (file_path, embedding) VALUES (%s, %s)",
            (str(file_path), vector)
        )

conn.commit()
cur.close()
conn.close()
print("Indexing complete!")