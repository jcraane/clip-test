CLIP Image Search & Embedding Service

This project provides a minimal end-to-end workflow for image search using OpenAI's CLIP model:
- A FastAPI microservice exposes a `/embed` endpoint that returns normalized text embeddings.
- A batch ingestion script computes CLIP embeddings for images and stores them in Postgres using the `pgvector` extension.
- A query script converts a text prompt to a vector and retrieves the nearest images from the database.
- A small demo labels script shows basic zero-shot image classification using CLIP.

What’s inside
- Model/ML: PyTorch + CLIP (`ViT-B/32`) with Apple Silicon acceleration via MPS when available.
- API: FastAPI (`embed_service.py`) served with Uvicorn.
- Storage/Vector Search: Postgres + `pgvector` (stores embeddings and performs nearest neighbor search).
- Data processing: Pillow (PIL) for image loading, `tqdm` for progress bars.
- DB client: `psycopg2`.

Key files
- `embed_service.py`: FastAPI app with POST `/embed` to convert text to a normalized embedding.
- `image_ingestion.py`: Recursively scans an image directory, computes CLIP image embeddings, and inserts `(file_path, embedding)` into Postgres (`image_library` table).
- `query_image.py`: Turns a text query into a vector and returns the top matches from Postgres using `pgvector` distance; optionally opens the top match on macOS.
- `image_labels.py`: Simple zero-shot example that scores candidate labels for a given image using CLIP.

Prerequisites
- Python 3.10+ (use a virtual environment)
- PyTorch with MPS support (on Apple Silicon) or CPU fallback
- Postgres with the `pgvector` extension enabled and a database configured (see scripts for connection params)

Quick start (from the project root, the folder containing `embed_service.py`)

Environment
```bash
source .venv/bin/activate
python -V
which python
which uvicorn
```

Run the embedding API
```bash
python -m uvicorn embed_service:app --host 0.0.0.0 --port 8001 --reload
```

Ingest images into Postgres (example)
```bash
# Adjust IMAGE_DIR and DB_PARAMS inside image_ingestion.py to match your setup
python image_ingestion.py
```

Query images by text
```bash
python query_image.py
# Then enter a prompt like: a beach at sunset
```

Notes
- The scripts auto-detect `mps` on macOS; otherwise they fall back to CPU.
- Ensure your Postgres instance has `pgvector` installed and the `image_library` table created with a `vector` column compatible with CLIP’s output dimensions (e.g., 512 for ViT-B/32).