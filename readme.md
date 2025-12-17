From the project root (the folder containing embed_service.py):

```bash
source .venv/bin/activate
python -V
which python
which uvicorn
```

```bash
python -m uvicorn embed_service:app --host 0.0.0.0 --port 8001 --reload
```