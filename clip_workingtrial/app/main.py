from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
import torch
import clip
from io import BytesIO
import torch.nn.functional as F
import pickle

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