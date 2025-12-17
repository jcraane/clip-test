import torch
import clip
from PIL import Image
import urllib.request
import shutil

# 1. Setup Device (MPS for Mac)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load Model
print("Loading model...")
model, preprocess = clip.load("ViT-B/32", device=device)

# 3. Download a sample image (User-Agent fix added)
# url = "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"
image_filename = "poek.jpg"
# print(f"Downloading sample image from {url}...")

# Create a request that mimics a browser (Mozilla)
# req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

# Download and save the file
# with urllib.request.urlopen(req) as response, open(image_filename, 'wb') as out_file:
#     shutil.copyfileobj(response, out_file)

# 4. Prepare the Image
image = preprocess(Image.open(image_filename)).unsqueeze(0).to(device)

# 5. Define the text labels you want CLIP to choose from
possible_labels = ["a diagram", "a dog", "a cat", "a photo of an astronaut", "A rat"]
text = clip.tokenize(possible_labels).to(device)

# 6. Run Inference
print("Analyzing image...")
with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 7. Print Results
print("\nResults:")
for label, probability in zip(possible_labels, probs[0]):
    print(f"{label}: {probability * 100:.2f}%")