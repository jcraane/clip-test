import os
import torch
import clip
from PIL import Image

# Optional: use CIFAR-100 class names for zero-shot labels (like the example)
try:
    from torchvision.datasets import CIFAR100
except Exception:
    CIFAR100 = None

# Optional: use ImageNet-1K class names (from torchvision weights metadata)
try:
    from torchvision.models import ResNet50_Weights
except Exception:
    ResNet50_Weights = None


def get_device():
    # Prefer Apple MPS on macOS, else CUDA if available, else CPU
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_labels(labelset: str = "cifar100"):
    """Load zero-shot label set.

    labelset:
      - "cifar100": CIFAR-100 class names (via torchvision.datasets.CIFAR100)
      - "imagenet1k": ImageNet-1K class names (via torchvision model weights metadata)
    Falls back to a small generic set if the requested labelset is unavailable.
    """
    labelset = (labelset or "").strip().lower()

    if labelset in {"imagenet1k", "imagenet-1k", "imagenet"}:
        if ResNet50_Weights is not None:
            try:
                weights = ResNet50_Weights.IMAGENET1K_V2
                categories = weights.meta.get("categories")
                if categories:
                    return list(categories)
            except Exception:
                pass

    if labelset in {"cifar100", "cifar-100", "cifar"}:
        # Try to read labels without downloading the dataset
        if CIFAR100 is not None:
            try:
                # Some torchvision versions expose classes at the class-level
                classes = getattr(CIFAR100, "classes", None)
                if classes:
                    return list(classes)

                # Otherwise, instantiate with download allowed to obtain metadata
                ds = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
                return list(ds.classes)
            except Exception:
                pass

    # Fallback labels (small demo set)
    return [
        "a dog",
        "a cat",
        "a bird",
        "a car",
        "a plane",
        "a ship",
        "a person",
        "a horse",
        "a cow",
        "a bicycle",
    ]


def main():
    device = get_device()
    print(f"Using device: {device}")

    print("Loading CLIP model (ViT-B/32)...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_path = os.path.join(os.path.dirname(__file__), "poek.jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Prepare inputs
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Choose labelset: "cifar100" (default) or "imagenet1k"
    # labelset = os.environ.get("LABELSET", "cifar100")
    labelset = os.environ.get("LABELSET", "imagenet1k")
    labels = load_labels(labelset=labelset)
    print(f"Using labelset: {labelset} ({len(labels)} labels)")

    # Prefix like in the example: "a photo of a {c}"
    prompts = [f"a photo of a {c}" for c in labels]
    text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    # Normalize and compute similarities
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    # Print results
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{labels[index]:>16s}: {100 * value.item():.2f}%")


if __name__ == "__main__":
    main()
