from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import clip
from io import BytesIO
import torch.nn.functional as F
import pickle

from app.database import init_db, insert_image, get_all_embeddings, insert_similarity, get_all_images

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_embedding(image_data):
    image = preprocess(Image.open(BytesIO(image_data))).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu()

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        embedding = get_image_embedding(image_data)

        # Convert embedding to bytes (to store in SQLite)
        embedding_bytes = pickle.dumps(embedding.tolist())
        filename = file.filename

        # Insert image into DB
        insert_image(filename, image_data, embedding_bytes)

        # Compute similarity with all existing images
        all_rows = get_all_embeddings()
        for img_id, img_name, img_embedding_bytes in all_rows:
            if img_name == filename:
                continue
            existing_embedding = torch.tensor(pickle.loads(img_embedding_bytes))
            similarity = F.cosine_similarity(embedding, existing_embedding).item()
            insert_similarity(img_id, None, similarity)  # store comparison record

        return {"filename": filename, "message": "Image uploaded and stored successfully."}

    except Exception as e:
        return {"error": f"An error occurred during upload: {e}"}

# Route to calculate and return similarity score
@app.post("/compare-images/")
async def compare_images(file: UploadFile = File(...)):
    try:
        rows = get_all_embeddings()
        if not rows:
            return {"error": "No images in database to compare against."}

        image_data = await file.read()
        new_embedding = get_image_embedding(image_data)

        best_match = None
        highest_score = -1

        for img_id, img_name, img_embedding_bytes in rows:
            existing_embedding = torch.tensor(pickle.loads(img_embedding_bytes))
            similarity = F.cosine_similarity(new_embedding, existing_embedding).item()
            if similarity > highest_score:
                highest_score = similarity
                best_match = img_name

        return {"best_match": best_match, "similarity_score": highest_score}

    except Exception as e:
        return {"error": f"An error occurred during comparison: {e}"}
    
@app.get("/get-images/")
def get_images():
    try:
        return get_all_images()
    except Exception as e:
        return {"error": f"Failed to fetch images: {e}"}