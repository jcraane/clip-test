import torch
import clip

# Check for MPS (Apple Silicon GPU)
if torch.backends.mps.is_available():
    device = "mps"
    print("✅ Success: M1 GPU (MPS) is available!")
else:
    device = "cpu"
    print("⚠️ Warning: Running on CPU (Slower)")

model, preprocess = clip.load("ViT-B/32", device=device)
print("CLIP loaded successfully.")