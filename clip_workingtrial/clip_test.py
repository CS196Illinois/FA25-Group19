import os
import torch
import clip
from PIL import Image
import torch.nn.functional as F

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Folder where your clothing images are stored
image_folder = "images"

# Step 1: Turn each image into an "embedding vector"
embeddings = {}
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_folder, filename)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        embeddings[filename] = image_features
        print(f"✅ Got embedding for {filename}")

# Step 2: Compare similarity between two items
filenames = list(embeddings.keys())
if len(filenames) >= 2:
    img1, img2 = filenames[0], filenames[1]
    sim = F.cosine_similarity(embeddings[img1], embeddings[img2]).item()
    print(f"\n🧢 Similarity between '{img1}' and '{img2}': {sim:.3f}")
else:
    print("Upload at least two clothing images to compare.")
