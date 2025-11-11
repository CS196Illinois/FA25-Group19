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

@app.post("/generate-picked-outfit/")
async def generate_picked_outfit(request: dict):
    """
    Generate an outfit based on a user-selected item (top or bottom).
    Finds the best matching item from the opposite category.
    
    Expected request body:
    {
        "item_id": int,
        "item_category": "top" | "bottom"
    }
    """
    try:
        item_id = request.get("item_id")
        item_category = request.get("item_category")
        
        # Validate inputs
        if not item_id or not item_category:
            raise HTTPException(status_code=400, detail="item_id and item_category are required")
        
        if item_category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="item_category must be 'top' or 'bottom'")
        
        # Get the selected item's data
        selected_item_data = get_image_by_id(item_id, item_category)
        if selected_item_data is None:
            raise HTTPException(status_code=404, detail=f"Selected {item_category} not found")
        
        # selected_item_data is tuple: (id, filename, image_data, embedding)
        selected_embedding = torch.tensor(pickle.loads(selected_item_data[3]))
        
        # Determine the opposite category
        opposite_category = 'bottom' if item_category == 'top' else 'top'
        
        # Get all items from the opposite category
        opposite_items = get_all_images(opposite_category)
        if not opposite_items:
            raise HTTPException(status_code=404, detail=f"No {opposite_category}s available in the database")
        
        # Find the item with highest similarity
        best_match = None
        highest_similarity = -1
        
        opposite_embeddings = get_all_embeddings(opposite_category)
        for opposite_id, opposite_filename, opposite_embedding_bytes in opposite_embeddings:
            opposite_embedding = torch.tensor(pickle.loads(opposite_embedding_bytes))
            similarity = F.cosine_similarity(selected_embedding, opposite_embedding).item()
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = {
                    "id": opposite_id,
                    "filename": opposite_filename
                }
        
        if best_match is None:
            raise HTTPException(status_code=404, detail=f"Could not find matching {opposite_category}")
        
        # Get full details for best match
        best_match_full = next((item for item in opposite_items if item["id"] == best_match["id"]), None)
        
        # Construct response based on which item was selected
        if item_category == 'top':
            selected_item_info = {
                "id": selected_item_data[0],
                "filename": selected_item_data[1],
                "uploaded_at": next((item["uploaded_at"] for item in get_all_images('top') if item["id"] == item_id), None)
            }
            return {
                "top": selected_item_info,
                "bottom": {
                    "id": best_match_full["id"],
                    "filename": best_match_full["filename"],
                    "uploaded_at": best_match_full["uploaded_at"]
                },
                "similarity_score": highest_similarity
            }
        else:  # item_category == 'bottom'
            selected_item_info = {
                "id": selected_item_data[0],
                "filename": selected_item_data[1],
                "uploaded_at": next((item["uploaded_at"] for item in get_all_images('bottom') if item["id"] == item_id), None)
            }
            return {
                "top": {
                    "id": best_match_full["id"],
                    "filename": best_match_full["filename"],
                    "uploaded_at": best_match_full["uploaded_at"]
                },
                "bottom": selected_item_info,
                "similarity_score": highest_similarity
            }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))