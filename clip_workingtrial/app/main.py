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
from transformers import YolosImageProcessor, YolosForObjectDetection

from app.database import (
    init_db, insert_image, get_all_embeddings, insert_similarity,
    get_all_images, get_image_by_id, delete_image,
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

# Load the YOLOS fashion detection model
fashion_model_id = 'yainage90/fashion-object-detection-yolos-tiny'
fashion_processor = YolosImageProcessor.from_pretrained(fashion_model_id)
fashion_model = YolosForObjectDetection.from_pretrained(fashion_model_id).to(device)

def get_image_embedding(image_data):
    image = preprocess(Image.open(BytesIO(image_data))).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu()

def split_and_embed_outfit(image_data):
    """
    Split outfit image into top and bottom halves, generate embeddings for each.
    Uses simple geometric split: top 50% = top garment, bottom 50% = bottom garment.
    """
    # Open the outfit image
    outfit_image = Image.open(BytesIO(image_data))
    width, height = outfit_image.size

    # Crop top half (top garment region)
    top_crop = outfit_image.crop((0, 0, width, height // 2))

    # Crop bottom half (bottom garment region)
    bottom_crop = outfit_image.crop((0, height // 2, width, height))

    # Generate CLIP embeddings for each region
    # Convert crops back to bytes for processing
    top_buffer = BytesIO()
    top_crop.save(top_buffer, format='PNG')
    top_embedding = get_image_embedding(top_buffer.getvalue())

    bottom_buffer = BytesIO()
    bottom_crop.save(bottom_buffer, format='PNG')
    bottom_embedding = get_image_embedding(bottom_buffer.getvalue())

    return top_embedding, bottom_embedding

def detect_and_embed_outfit(image_data):
    """
    Use YOLOS to detect tops and bottoms in outfit image, then generate CLIP embeddings.
    More accurate than geometric splitting - detects actual garment regions.
    Falls back to geometric split if detection fails.
    """
    # Open the outfit image
    outfit_image = Image.open(BytesIO(image_data))

    # Run YOLOS fashion detection
    inputs = fashion_processor(images=[outfit_image], return_tensors="pt")
    outputs = fashion_model(**inputs.to(device))

    # Post-process detection results
    results = fashion_processor.post_process_object_detection(
        outputs,
        threshold=0.5,  # confidence threshold
        target_sizes=torch.tensor([[outfit_image.size[1], outfit_image.size[0]]])
    )[0]

    # Find best top and bottom detections
    best_top = None
    best_bottom = None

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        class_name = fashion_model.config.id2label[label.item()]
        confidence = score.item()

        if class_name == 'top':
            if best_top is None or confidence > best_top['conf']:
                best_top = {'box': box.tolist(), 'conf': confidence}
        elif class_name == 'bottom':
            if best_bottom is None or confidence > best_bottom['conf']:
                best_bottom = {'box': box.tolist(), 'conf': confidence}

    # If detection failed, fall back to geometric split
    if best_top is None or best_bottom is None:
        print("YOLOS detection failed, falling back to geometric split")
        return split_and_embed_outfit(image_data)

    # Crop detected top region
    x1, y1, x2, y2 = [int(coord) for coord in best_top['box']]
    top_crop = outfit_image.crop((x1, y1, x2, y2))
    top_buffer = BytesIO()
    top_crop.save(top_buffer, format='PNG')
    top_embedding = get_image_embedding(top_buffer.getvalue())

    # Crop detected bottom region
    x1, y1, x2, y2 = [int(coord) for coord in best_bottom['box']]
    bottom_crop = outfit_image.crop((x1, y1, x2, y2))
    bottom_buffer = BytesIO()
    bottom_crop.save(bottom_buffer, format='PNG')
    bottom_embedding = get_image_embedding(bottom_buffer.getvalue())

    return top_embedding, bottom_embedding

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...), category: str = Form(...)):
    try:
        # Validate category
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")

        image_data = await file.read()
        embedding = get_image_embedding(image_data)
        filename = file.filename

        # Convert embedding to bytes (to store in SQLite)
        embedding_bytes = pickle.dumps(embedding.tolist())

        # CHECK 1: Detect duplicate filename
        existing_images = get_all_images(category)
        for existing_img in existing_images:
            if existing_img["filename"] == filename:
                # Duplicate filename found - return 409 Conflict with details
                raise HTTPException(
                    status_code=409,
                    detail={
                        "type": "filename_duplicate",
                        "message": f"Filename '{filename}' already exists in {category}s",
                        "existing_id": existing_img["id"],
                        "existing_filename": existing_img["filename"]
                    }
                )

        # CHECK 2: Detect duplicate content using embedding similarity
        similarity_threshold = 0.95
        all_embeddings = get_all_embeddings(category)
        for existing_id, existing_filename, existing_embedding_bytes in all_embeddings:
            existing_embedding = torch.tensor(pickle.loads(existing_embedding_bytes))
            similarity = F.cosine_similarity(embedding, existing_embedding).item()

            if similarity > similarity_threshold:
                # Duplicate content found - return 409 Conflict with details
                raise HTTPException(
                    status_code=409,
                    detail={
                        "type": "content_duplicate",
                        "message": f"This image looks identical to '{existing_filename}' ({similarity*100:.1f}% similar)",
                        "existing_id": existing_id,
                        "existing_filename": existing_filename,
                        "similarity_score": round(similarity, 4)
                    }
                )

        # Both checks passed - insert image into DB with category
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

@app.post("/upload-outfit/")
async def upload_outfit(file: UploadFile = File(...)):
    """
    Upload a complete outfit image (person wearing top + bottom).
    Uses YOLOS to detect and crop top/bottom regions, stores as linked outfit.
    """
    try:
        image_data = await file.read()

        # Use YOLOS to detect and embed top and bottom
        top_embedding, bottom_embedding = detect_and_embed_outfit(image_data)

        # Convert embeddings to bytes
        top_embedding_bytes = pickle.dumps(top_embedding.tolist())
        bottom_embedding_bytes = pickle.dumps(bottom_embedding.tolist())

        # Insert into outfits table (keeps top and bottom linked)
        outfit_id = insert_outfit(image_data, top_embedding_bytes, bottom_embedding_bytes)

        return {
            "filename": file.filename,
            "message": "Outfit uploaded successfully - top and bottom detected and linked",
            "outfit_id": outfit_id
        }

    except Exception as e:
        return {"error": f"An error occurred during outfit upload: {e}"}

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

@app.get("/get-outfits/")
def get_outfits():
    """
    Get all complete outfits (metadata only, no image data)
    """
    try:
        return get_all_outfits()
    except Exception as e:
        return {"error": f"Failed to fetch outfits: {e}"}

@app.get("/get-outfit/{outfit_id}")
async def get_outfit(outfit_id: int):
    """
    Get complete outfit image by id
    """
    try:
        outfit_data = get_outfit_by_id(outfit_id)
        if outfit_data is None:
            raise HTTPException(status_code=404, detail="Outfit not found")

        # outfit_data is a tuple: (id, outfit_image_data, top_embedding, bottom_embedding, uploaded_at)
        # Return the outfit image binary data (index 1)
        return Response(content=outfit_data[1], media_type="image/jpeg")

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

        # Find top 5 items with highest similarity to selected item
        matches_with_scores = []

        opposite_embeddings = get_all_embeddings(opposite_category)
        for opposite_id, opposite_filename, opposite_embedding_bytes in opposite_embeddings:
            opposite_embedding = torch.tensor(pickle.loads(opposite_embedding_bytes))
            similarity = F.cosine_similarity(selected_embedding, opposite_embedding).item()

            matches_with_scores.append({
                "id": opposite_id,
                "filename": opposite_filename,
                "similarity": similarity,
                "embedding": opposite_embedding
            })

        # Sort by similarity descending and take top 5
        matches_with_scores.sort(key=lambda x: x["similarity"], reverse=True)
        top_5_matches = matches_with_scores[:5]

        if not top_5_matches:
            raise HTTPException(status_code=404, detail=f"Could not find matching {opposite_category}")

        # For each of the top 5 matches, find 3 variations (the match itself + 2 similar items)
        outfit_suggestions = []

        for main_match in top_5_matches:
            # Find items similar to this match (variations)
            variations = []

            for candidate_id, candidate_filename, candidate_embedding_bytes in opposite_embeddings:
                if candidate_id == main_match["id"]:
                    continue  # Skip the main match itself when finding variations

                candidate_embedding = torch.tensor(pickle.loads(candidate_embedding_bytes))
                variation_similarity = F.cosine_similarity(main_match["embedding"], candidate_embedding).item()

                variations.append({
                    "id": candidate_id,
                    "filename": candidate_filename,
                    "similarity_to_main": variation_similarity
                })

            # Sort variations and take top 2
            variations.sort(key=lambda x: x["similarity_to_main"], reverse=True)
            top_2_variations = variations[:2]

            # Add main match + 2 variations to suggestions
            outfit_variations = [
                {
                    "id": main_match["id"],
                    "filename": main_match["filename"],
                    "similarity_to_selected": main_match["similarity"]
                }
            ]

            for var in top_2_variations:
                outfit_variations.append({
                    "id": var["id"],
                    "filename": var["filename"],
                    "similarity_to_selected": main_match["similarity"]  # Use main match similarity
                })

            outfit_suggestions.extend(outfit_variations)

        # Get selected item info
        selected_item_info = {
            "id": selected_item_data[0],
            "filename": selected_item_data[1],
            "uploaded_at": next((item["uploaded_at"] for item in get_all_images(item_category) if item["id"] == item_id), None)
        }

        # Build list of all outfit suggestions with full details
        all_suggestions = []
        for suggestion in outfit_suggestions:
            match_full = next((item for item in opposite_items if item["id"] == suggestion["id"]), None)
            if match_full:
                all_suggestions.append({
                    "id": match_full["id"],
                    "filename": match_full["filename"],
                    "uploaded_at": match_full["uploaded_at"],
                    "similarity_score": suggestion["similarity_to_selected"]
                })

        # Return selected item + all suggestions
        return {
            "selected_item": selected_item_info,
            "selected_category": item_category,
            "suggestions": all_suggestions,
            "opposite_category": opposite_category
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
async def find_outfit_by_item(file: UploadFile = File(...), category: str = Form(...)):
    """
    Upload a single clothing item and get a recommendation from your closet.

    Workflow:
    1. Find an outfit with similar item in the specified category
    2. Get the opposite piece from that outfit (if you uploaded top, get outfit's bottom)
    3. Search YOUR closet for items similar to that outfit piece
    4. Return recommendation from your actual wardrobe

    Args:
        file: Image of a single clothing item (top or bottom)
        category: 'top' or 'bottom' - which type of clothing item this is

    Returns:
        Recommendation with item from user's closet and confidence level
    """
    try:
        # Validate category
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")

        # Get all outfit embeddings
        outfit_embeddings = get_all_outfit_embeddings()
        if not outfit_embeddings:
            raise HTTPException(status_code=404, detail="No outfits in database to search")

        # Generate embedding for the uploaded item
        item_image_data = await file.read()
        item_embedding = get_image_embedding(item_image_data)

        # STEP 1: Find top 5 best matching outfits for uploaded item
        outfit_matches = []

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

            outfit_matches.append({
                "outfit_id": outfit_id,
                "similarity": similarity,
                "opposite_embedding": torch.tensor(pickle.loads(opposite_embedding_bytes))
            })

        # Sort by similarity descending and take top 5 outfits
        outfit_matches.sort(key=lambda x: x["similarity"], reverse=True)
        top_5_outfits = outfit_matches[:5]

        if not top_5_outfits:
            raise HTTPException(status_code=404, detail="Could not find matching outfit")

        # STEP 2: For each of the 5 best outfits, find 3 best matching items from closet
        opposite_category = 'bottom' if category == 'top' else 'top'
        closet_items = get_all_embeddings(opposite_category)

        if not closet_items:
            raise HTTPException(
                status_code=404,
                detail=f"No {opposite_category}s in your closet. Upload some {opposite_category}s first!"
            )

        all_suggestions = []

        for outfit_match in top_5_outfits:
            # For this specific outfit, find best 3 closet items
            closet_matches = []

            for closet_item_id, closet_item_name, closet_embedding_bytes in closet_items:
                closet_embedding = torch.tensor(pickle.loads(closet_embedding_bytes))
                similarity = F.cosine_similarity(outfit_match["opposite_embedding"], closet_embedding).item()

                closet_matches.append({
                    "id": closet_item_id,
                    "filename": closet_item_name,
                    "similarity": similarity,
                    "outfit_id": outfit_match["outfit_id"]
                })

            # Sort by similarity and take top 3 for this outfit
            closet_matches.sort(key=lambda x: x["similarity"], reverse=True)
            top_3_closet_items = closet_matches[:3]

            # Add these 3 suggestions
            for closet_item in top_3_closet_items:
                all_suggestions.append({
                    "id": closet_item["id"],
                    "filename": closet_item["filename"],
                    "similarity_to_outfit": closet_item["similarity"],
                    "reference_outfit_id": closet_item["outfit_id"]
                })

        # Determine confidence level based on best match
        best_similarity = all_suggestions[0]["similarity_to_outfit"] if all_suggestions else 0
        confidence = "high" if best_similarity > 0.6 else "low"
        confidence_message = ""
        if confidence == "low":
            confidence_message = f"These are the closest matches from your closet ({best_similarity*100:.1f}%), but they're not very similar to the styled outfit."

        return {
            "message": f"Based on outfits with similar {category}s, here are suggestions from your closet",
            "outfit_reference_id": top_5_outfits[0]["outfit_id"],  # First/best outfit
            "outfit_similarity": top_5_outfits[0]["similarity"],
            "uploaded_item_category": category,
            "opposite_category": opposite_category,
            "suggestions": all_suggestions,
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