from fastapi import FastAPI
from pydantic import BaseModel
import torch
import clip

app = FastAPI()

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: list[float]  # normalized

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    text_tokens = clip.tokenize([req.text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    vec = text_features.cpu().numpy().flatten().tolist()
    return {"embedding": vec}