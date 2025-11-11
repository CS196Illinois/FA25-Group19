from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
import torch
import clip
from io import BytesIO
import torch.nn.functional as F
import pickle
import random

from app.database import (
    init_db, insert_image, get_all_embeddings, insert_similarity,
    get_all_images, get_image_by_id, delete_image
)

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
async def upload_image(file: UploadFile = File(...), category: str = Form(...)):
    try:
        # Validate category
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")
        
        image_data = await file.read()
        embedding = get_image_embedding(image_data)

        # Convert embedding to bytes (to store in SQLite)
        embedding_bytes = pickle.dumps(embedding.tolist())
        filename = file.filename

        # Insert image into DB with category
        image_id = insert_image(filename, image_data, embedding_bytes, category)

        # Compute similarity with all existing images in the same category
        all_rows = get_all_embeddings(category)
        for img_id, img_name, img_embedding_bytes in all_rows:
            if img_id == image_id:  # Skip comparing with itself
                continue
            existing_embedding = torch.tensor(pickle.loads(img_embedding_bytes))
            similarity = F.cosine_similarity(embedding, existing_embedding).item()
            
            # Store similarity based on category
            if category == 'top':
                insert_similarity(image_id, img_id, similarity)
            else:
                insert_similarity(img_id, image_id, similarity)

        return {
            "filename": filename, 
            "message": f"Image uploaded successfully to {category}s",
            "id": image_id,
            "category": category
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        return {"error": f"An error occurred during upload: {e}"}

# Route to calculate and return similarity score
@app.post("/compare-images/")
async def compare_images(file: UploadFile = File(...), category: str = Form(...)):
    try:
        # Validate category
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")
        
        rows = get_all_embeddings(category)
        if not rows:
            return {"error": f"No {category} images in database to compare against."}

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

        return {"best_match": best_match, "similarity_score": highest_score, "category": category}

    except HTTPException as he:
        raise he
    except Exception as e:
        return {"error": f"An error occurred during comparison: {e}"}

@app.get("/get-images/{category}")
def get_images(category: str):
    try:
        # Validate category
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")
        
        return get_all_images(category)
    except HTTPException as he:
        raise he
    except Exception as e:
        return {"error": f"Failed to fetch images: {e}"}

@app.get("/get-image/{category}/{image_id}")
async def get_image(category: str, image_id: int):
    """
    Get image data by id and return as binary image
    """
    try:
        # Validate category
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")
        
        image_data = get_image_by_id(image_id, category)
        if image_data is None:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # image_data is a tuple: (id, filename, image_data, embedding)
        # Return the image binary data (index 2)
        return Response(content=image_data[2], media_type="image/jpeg")
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-image/{category}/{image_id}")
async def delete_image_endpoint(category: str, image_id: int):
    """
    Delete an image by id from the specified category
    """
    try:
        # Validate category
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")

        delete_image(image_id, category)
        return {"message": f"Image {image_id} deleted successfully from {category}s"}

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate-random-outfit/")
async def generate_random_outfit():
    """
    Generate a random outfit by selecting one random top and one random bottom
    """
    try:
        # Get all tops and bottoms
        tops = get_all_images('top')
        bottoms = get_all_images('bottom')

        # Handle edge cases
        if not tops and not bottoms:
            raise HTTPException(status_code=404, detail="No tops or bottoms available in the database")

        if not tops:
            raise HTTPException(status_code=404, detail="No tops available in the database")

        if not bottoms:
            raise HTTPException(status_code=404, detail="No bottoms available in the database")

        # Randomly select one top and one bottom
        random_top = random.choice(tops)
        random_bottom = random.choice(bottoms)

        return {
            "top": {
                "id": random_top["id"],
                "filename": random_top["filename"],
                "uploaded_at": random_top["uploaded_at"]
            },
            "bottom": {
                "id": random_bottom["id"],
                "filename": random_bottom["filename"],
                "uploaded_at": random_bottom["uploaded_at"]
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate-similarity-outfit/")
async def generate_similarity_outfit():
    """
    Generate an outfit by selecting a random top and finding the bottom with highest similarity
    """
    try:
        # Get all tops and bottoms
        tops = get_all_images('top')
        bottoms = get_all_images('bottom')

        # Handle edge cases
        if not tops and not bottoms:
            raise HTTPException(status_code=404, detail="No tops or bottoms available in the database")

        if not tops:
            raise HTTPException(status_code=404, detail="No tops available in the database")

        if not bottoms:
            raise HTTPException(status_code=404, detail="No bottoms available in the database")

        # Randomly select one top
        random_top = random.choice(tops)

        # Get the top's embedding
        top_data = get_image_by_id(random_top["id"], 'top')
        if top_data is None:
            raise HTTPException(status_code=404, detail="Selected top not found")

        # top_data is tuple: (id, filename, image_data, embedding)
        top_embedding = torch.tensor(pickle.loads(top_data[3]))

        # Find the bottom with highest similarity to the selected top
        best_bottom = None
        highest_similarity = -1

        bottom_embeddings = get_all_embeddings('bottom')
        for bottom_id, bottom_filename, bottom_embedding_bytes in bottom_embeddings:
            bottom_embedding = torch.tensor(pickle.loads(bottom_embedding_bytes))
            similarity = F.cosine_similarity(top_embedding, bottom_embedding).item()

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_bottom = {
                    "id": bottom_id,
                    "filename": bottom_filename
                }

        # Get the full bottom details
        if best_bottom is None:
            raise HTTPException(status_code=404, detail="Could not find matching bottom")

        # Find the bottom in the bottoms list to get uploaded_at
        best_bottom_full = next((b for b in bottoms if b["id"] == best_bottom["id"]), None)

        return {
            "top": {
                "id": random_top["id"],
                "filename": random_top["filename"],
                "uploaded_at": random_top["uploaded_at"]
            },
            "bottom": {
                "id": best_bottom_full["id"],
                "filename": best_bottom_full["filename"],
                "uploaded_at": best_bottom_full["uploaded_at"]
            },
            "similarity_score": highest_similarity
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))