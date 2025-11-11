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
    get_all_images, get_image_by_id, delete_image, update_temperature_tag,
    insert_outfit, get_all_outfits, get_outfit_by_id, get_all_outfit_embeddings
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

# Temperature category ranges (in Fahrenheit)
TEMP_CATEGORIES = {
    'cold': (float('-inf'), 50),
    'cool': (50, 65),
    'mild': (65, 75),
    'warm': (75, 85),
    'hot': (85, float('inf'))
}

# Order of temperature categories for fallback logic
TEMP_ORDER = ['cold', 'cool', 'mild', 'warm', 'hot']

def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit"""
    return (celsius * 9/5) + 32

def get_temperature_category(temp_fahrenheit):
    """Get temperature category for a given temperature in Fahrenheit"""
    for category, (min_temp, max_temp) in TEMP_CATEGORIES.items():
        if min_temp < temp_fahrenheit <= max_temp:
            return category
    return None

def get_fallback_category(target_category, avg_temp):
    """
    Get the next closest temperature category if target category has no matches.
    Logic: If avg_temp is closer to the upper bound, try the warmer category first,
    otherwise try the cooler category first.
    """
    if target_category not in TEMP_ORDER:
        return []

    current_index = TEMP_ORDER.index(target_category)
    min_temp, max_temp = TEMP_CATEGORIES[target_category]

    # Determine which direction to search first based on where avg_temp falls in range
    if max_temp != float('inf') and min_temp != float('-inf'):
        mid_point = (min_temp + max_temp) / 2
        if avg_temp >= mid_point:
            # Closer to upper bound, try warmer first
            fallback_order = []
            # Add warmer categories
            for i in range(current_index + 1, len(TEMP_ORDER)):
                fallback_order.append(TEMP_ORDER[i])
            # Add cooler categories
            for i in range(current_index - 1, -1, -1):
                fallback_order.append(TEMP_ORDER[i])
        else:
            # Closer to lower bound, try cooler first
            fallback_order = []
            # Add cooler categories
            for i in range(current_index - 1, -1, -1):
                fallback_order.append(TEMP_ORDER[i])
            # Add warmer categories
            for i in range(current_index + 1, len(TEMP_ORDER)):
                fallback_order.append(TEMP_ORDER[i])
    elif max_temp == float('inf'):
        # Hot category - only go cooler
        fallback_order = [TEMP_ORDER[i] for i in range(current_index - 1, -1, -1)]
    else:
        # Cold category - only go warmer
        fallback_order = [TEMP_ORDER[i] for i in range(current_index + 1, len(TEMP_ORDER))]

    return fallback_order

def get_image_embedding(image_data):
    image = preprocess(Image.open(BytesIO(image_data))).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu()

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...), category: str = Form(...), temperature_tag: str = Form(None)):
    try:
        # Validate category
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")

        # Validate temperature_tag if provided
        valid_temp_tags = ['cold', 'cool', 'mild', 'warm', 'hot']
        if temperature_tag and temperature_tag not in valid_temp_tags:
            raise HTTPException(status_code=400, detail=f"Temperature tag must be one of: {', '.join(valid_temp_tags)}")

        image_data = await file.read()
        embedding = get_image_embedding(image_data)

        # Convert embedding to bytes (to store in SQLite)
        embedding_bytes = pickle.dumps(embedding.tolist())
        filename = file.filename

        # Insert image into DB with category and temperature_tag
        image_id = insert_image(filename, image_data, embedding_bytes, category, temperature_tag)

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

@app.patch("/update-temperature-tag/{category}/{image_id}")
async def update_temperature_tag_endpoint(category: str, image_id: int, temperature_tag: str = Form(...)):
    """
    Update the temperature tag for an existing image
    """
    try:
        # Validate category
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")

        # Validate temperature_tag
        valid_temp_tags = ['cold', 'cool', 'mild', 'warm', 'hot']
        if temperature_tag not in valid_temp_tags:
            raise HTTPException(status_code=400, detail=f"Temperature tag must be one of: {', '.join(valid_temp_tags)}")

        update_temperature_tag(image_id, category, temperature_tag)
        return {"message": f"Temperature tag updated to '{temperature_tag}' for image {image_id}"}

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

@app.post("/upload-outfit/")
async def upload_outfit(file: UploadFile = File(...)):
    """
    Upload a complete outfit image (photo of person wearing full outfit).
    The image will be split into top/bottom halves and embeddings generated for each.
    """
    try:
        outfit_image_data = await file.read()

        # Split image and generate embeddings for top and bottom regions
        top_embedding, bottom_embedding = split_and_embed_outfit(outfit_image_data)

        # Convert embeddings to bytes for storage
        top_embedding_bytes = pickle.dumps(top_embedding.tolist())
        bottom_embedding_bytes = pickle.dumps(bottom_embedding.tolist())

        # Store outfit in database
        outfit_id = insert_outfit(outfit_image_data, top_embedding_bytes, bottom_embedding_bytes)

        return {
            "message": "Outfit uploaded successfully",
            "outfit_id": outfit_id,
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload outfit: {str(e)}")

@app.post("/find-outfit-by-item/")
async def find_outfit_by_item(
    file: UploadFile = File(...),
    category: str = Form(...),
    low_temp: float = Form(None),
    high_temp: float = Form(None),
    temp_unit: str = Form(None)
):
    """
    Upload a single clothing item and get a recommendation from your closet.

    Workflow:
    1. Find an outfit with similar item in the specified category
    2. Get the opposite piece from that outfit (if you uploaded top, get outfit's bottom)
    3. Search YOUR closet for items similar to that outfit piece
    4. Filter by temperature category if temperature range provided
    5. Return recommendation from your actual wardrobe

    Args:
        file: Image of a single clothing item (top or bottom)
        category: 'top' or 'bottom' - which type of clothing item this is
        low_temp: Lowest temperature of the day (optional)
        high_temp: Highest temperature of the day (optional)
        temp_unit: 'fahrenheit' or 'celsius' (optional, required if temps provided)

    Returns:
        Recommendation with item from user's closet and confidence level
    """
    try:
        # Validate category
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")

        # Validate temperature inputs
        target_temp_category = None
        avg_temp_fahrenheit = None
        if low_temp is not None or high_temp is not None:
            if low_temp is None or high_temp is None or temp_unit is None:
                raise HTTPException(status_code=400, detail="Must provide low_temp, high_temp, and temp_unit together")

            if temp_unit not in ['fahrenheit', 'celsius']:
                raise HTTPException(status_code=400, detail="temp_unit must be 'fahrenheit' or 'celsius'")

            # Convert to Fahrenheit if needed
            if temp_unit == 'celsius':
                low_temp_f = celsius_to_fahrenheit(low_temp)
                high_temp_f = celsius_to_fahrenheit(high_temp)
            else:
                low_temp_f = low_temp
                high_temp_f = high_temp

            # Calculate average temperature
            avg_temp_fahrenheit = (low_temp_f + high_temp_f) / 2

            # Get temperature category
            target_temp_category = get_temperature_category(avg_temp_fahrenheit)
            if not target_temp_category:
                raise HTTPException(status_code=400, detail="Invalid temperature range")

        # Get all outfit embeddings
        outfit_embeddings = get_all_outfit_embeddings()
        if not outfit_embeddings:
            raise HTTPException(status_code=404, detail="No outfits in database to search")

        # Generate embedding for the uploaded item
        item_image_data = await file.read()
        item_embedding = get_image_embedding(item_image_data)

        # STEP 1: Search through all outfits to find best match for uploaded item
        best_outfit_id = None
        highest_outfit_similarity = -1
        best_outfit_opposite_embedding = None

        for outfit_id, top_embedding_bytes, bottom_embedding_bytes in outfit_embeddings:
            # Choose which embedding to compare based on category
            if category == 'top':
                outfit_embedding = torch.tensor(pickle.loads(top_embedding_bytes))
                opposite_embedding_bytes = bottom_embedding_bytes
            else:  # category == 'bottom'
                outfit_embedding = torch.tensor(pickle.loads(bottom_embedding_bytes))
                opposite_embedding_bytes = top_embedding_bytes

            # Compute similarity
            similarity = F.cosine_similarity(item_embedding, outfit_embedding).item()

            if similarity > highest_outfit_similarity:
                highest_outfit_similarity = similarity
                best_outfit_id = outfit_id
                best_outfit_opposite_embedding = torch.tensor(pickle.loads(opposite_embedding_bytes))

        if best_outfit_id is None:
            raise HTTPException(status_code=404, detail="Could not find matching outfit")

        # STEP 2: Now search user's closet for items similar to the outfit's opposite piece
        opposite_category = 'bottom' if category == 'top' else 'top'

        # If temperature filtering is enabled, filter by temperature category
        if target_temp_category:
            # Try target category first
            all_closet_items = get_all_images(opposite_category)
            filtered_items = [item for item in all_closet_items if item['temperature_tag'] == target_temp_category]

            # If no items found, try fallback categories
            if not filtered_items:
                fallback_categories = get_fallback_category(target_temp_category, avg_temp_fahrenheit)
                for fallback_cat in fallback_categories:
                    filtered_items = [item for item in all_closet_items if item['temperature_tag'] == fallback_cat]
                    if filtered_items:
                        break

            # If still no items found, return error message
            if not filtered_items:
                raise HTTPException(
                    status_code=404,
                    detail=f"No {opposite_category}s in your closet appropriate for the weather. Try uploading more variety of clothes!"
                )

            # Get embeddings only for filtered items
            closet_items = get_all_embeddings(opposite_category)
            filtered_ids = set(item['id'] for item in filtered_items)
            closet_items = [(item_id, item_name, emb) for item_id, item_name, emb in closet_items if item_id in filtered_ids]
        else:
            # No temperature filtering - use all items
            closet_items = get_all_embeddings(opposite_category)

        if not closet_items:
            raise HTTPException(
                status_code=404,
                detail=f"No {opposite_category}s in your closet. Upload some {opposite_category}s first!"
            )

        # Find best match from user's closet
        best_closet_item_id = None
        best_closet_item_name = None
        highest_closet_similarity = -1

        for closet_item_id, closet_item_name, closet_embedding_bytes in closet_items:
            closet_embedding = torch.tensor(pickle.loads(closet_embedding_bytes))
            similarity = F.cosine_similarity(best_outfit_opposite_embedding, closet_embedding).item()

            if similarity > highest_closet_similarity:
                highest_closet_similarity = similarity
                best_closet_item_id = closet_item_id
                best_closet_item_name = closet_item_name

        if best_closet_item_id is None:
            raise HTTPException(status_code=404, detail=f"Could not find matching {opposite_category} in your closet")

        # STEP 3: Determine confidence level
        confidence = "high" if highest_closet_similarity > 0.6 else "low"
        confidence_message = ""
        if confidence == "low":
            confidence_message = f"This is the closest match from your closet ({highest_closet_similarity*100:.1f}%), but it's not very similar to the styled outfit."

        # Get full closet item data
        closet_item_data = get_image_by_id(best_closet_item_id, opposite_category)

        return {
            "message": f"Based on outfits with similar {category}s, here's what to pair it with from your closet",
            "outfit_reference_id": best_outfit_id,
            "outfit_similarity": highest_outfit_similarity,
            "uploaded_item_category": category,
            "recommended_item": {
                "id": best_closet_item_id,
                "filename": best_closet_item_name,
                "category": opposite_category,
                "similarity_to_outfit": highest_closet_similarity
            },
            "confidence": confidence,
            "confidence_message": confidence_message
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find outfit: {str(e)}")

@app.get("/get-outfits/")
def get_outfits():
    """
    Get list of all uploaded outfits (metadata only, no image data).
    """
    try:
        return get_all_outfits()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch outfits: {str(e)}")

@app.get("/get-outfit/{outfit_id}")
async def get_outfit(outfit_id: int):
    """
    Get a specific outfit image by ID and return as binary image.
    """
    try:
        outfit_data = get_outfit_by_id(outfit_id)
        if outfit_data is None:
            raise HTTPException(status_code=404, detail="Outfit not found")

        # outfit_data is tuple: (id, outfit_image_data, top_embedding, bottom_embedding, uploaded_at)
        # Return the outfit image binary data (index 1)
        return Response(content=outfit_data[1], media_type="image/jpeg")

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))