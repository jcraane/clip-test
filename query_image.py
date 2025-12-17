import torch
import clip
import psycopg2
from tqdm import tqdm
from PIL import Image
import subprocess  # <-- add
import os          # <-- add

# 1. Database Connection Config (Matching your Docker setup)
DB_PARAMS = {
    "dbname": "embeddings",
    "user": "postgres",  # <-- Update these
    "password": "postgres",  # <-- Update these
    "host": "localhost",
    "port": "5441"
}

# 2. Load CLIP (Uses M1 GPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading CLIP model on {device}...")
model, _ = clip.load("ViT-B/32", device=device)

def _to_pgvector_literal(vec):
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def search_images(query_text, top_k=5):
    """Converts text to vector and queries Postgres for nearest images."""
    print(f"Searching for: '{query_text}'")
    text_tokens = clip.tokenize([query_text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    query_vector = text_features.cpu().numpy().flatten().tolist()
    query_vector_lit = _to_pgvector_literal(query_vector)

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        query = """
                SELECT file_path, 1 - (embedding <=> %s::vector) AS similarity
                FROM image_library
                ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """

        cur.execute(query, (query_vector_lit, query_vector_lit, top_k))
        results = cur.fetchall()

        cur.close()
        conn.close()
        return results

    except Exception as e:
        print(f"❌ Database error: {e}")
        return []


def display_top_match(matches):
    """Opens the highest-scoring image (first row) using macOS 'open'."""
    if not matches:
        print("No results to display.")
        return

    top_path, top_score = matches[0]
    print(f"\nShowing top match: [{top_score:.2%}] {top_path}")

    top_path = os.path.expanduser(top_path)

    if not os.path.exists(top_path):
        print(f"File does not exist: {top_path}")
        return

    # Open with the default associated app (usually Preview for images)
    subprocess.run(["open", top_path], check=False)


# --- Example Usage ---
if __name__ == "__main__":
    search_term = input("Enter what you are looking for (e.g. 'a beach'): ")
    matches = search_images(search_term)

    print("\n--- Top Matches ---")
    if not matches:
        print("No results found.")
    else:
        for path, score in matches:
            print(f"[{score:.2%} Match] {path}")

        display_top_match(matches)