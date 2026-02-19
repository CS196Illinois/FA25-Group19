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
    get_all_images, get_image_by_id, delete_image, rename_image,
    insert_outfit, get_all_outfits, get_outfit_by_id, get_all_outfit_embeddings, delete_outfit
)

app = FastAPI()

# configure cors
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

# initialize database
init_db()

# load clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_embedding(image_data):
    image = preprocess(Image.open(BytesIO(image_data))).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu()

def split_and_embed_outfit(image_data):
    """
    split outfit image into top and bottom halves, generate embeddings for each.
    uses simple geometric split: top 50% = top garment, bottom 50% = bottom garment.
    """
    # open outfit image
    outfit_image = Image.open(BytesIO(image_data))
    width, height = outfit_image.size

    # crop top half
    top_crop = outfit_image.crop((0, 0, width, height // 2))

    # crop bottom half
    bottom_crop = outfit_image.crop((0, height // 2, width, height))

    # generate clip embeddings for each region
    # convert crops back to bytes for processing
    top_buffer = BytesIO()
    top_crop.save(top_buffer, format='PNG')
    top_embedding = get_image_embedding(top_buffer.getvalue())

    bottom_buffer = BytesIO()
    bottom_crop.save(bottom_buffer, format='PNG')
    bottom_embedding = get_image_embedding(bottom_buffer.getvalue())

    return top_embedding, bottom_embedding

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...), category: str = Form(...), skip_content_check: bool = Form(False)):
    try:
        # validate category as top or bottom
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")

        image_data = await file.read()
        embedding = get_image_embedding(image_data)
        filename = file.filename

        # convert embedding to bytes (to store in sqlite)
        embedding_bytes = pickle.dumps(embedding.tolist())

        # check 1: detect duplicate filename
        existing_images = get_all_images(category)
        for existing_img in existing_images:
            if existing_img["filename"] == filename:
                # if duplicate filename found, return 409 conflict with details
                raise HTTPException(
                    status_code=409,
                    detail={
                        "type": "filename_duplicate",
                        "message": f"Filename '{filename}' already exists in {category}s",
                        "existing_id": existing_img["id"],
                        "existing_filename": existing_img["filename"]
                    }
                )

        # check 2: detect duplicate content using embedding similarity
        # skip this if user explicitly chose to add separately
        if not skip_content_check:
            similarity_threshold = 0.95
            all_embeddings = get_all_embeddings(category)
            for existing_id, existing_filename, existing_embedding_bytes in all_embeddings:
                existing_embedding = torch.tensor(pickle.loads(existing_embedding_bytes))
                similarity = F.cosine_similarity(embedding, existing_embedding).item()

                if similarity > similarity_threshold:
                    # duplicate content found - return 409 conflict with details
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

        # both checks passed -> insert image into db with category
        image_id = insert_image(filename, image_data, embedding_bytes, category)

        # compute similarity with all existing images in the same category
        all_rows = get_all_embeddings(category)
        for img_id, img_name, img_embedding_bytes in all_rows:
            if img_id == image_id:  # skip comparing with itself
                continue
            existing_embedding = torch.tensor(pickle.loads(img_embedding_bytes))
            similarity = F.cosine_similarity(embedding, existing_embedding).item()

            # store similarity based on category
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


# route to calculate and return similarity score
@app.post("/compare-images/")
async def compare_images(file: UploadFile = File(...), category: str = Form(...)):
    try:
        # validate category
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
        # validate category
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
    get image data by id and return as binary image
    """
    try:
        # validate category
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")
        
        image_data = get_image_by_id(image_id, category)
        if image_data is None:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # image_data is a tuple: (id, filename, image_data, embedding)
        # return image binary data (index 2)
        return Response(content=image_data[2], media_type="image/jpeg")
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-image/{category}/{image_id}")
async def delete_image_endpoint(category: str, image_id: int):
    """
    delete an image by id from the specified category
    """
    try:
        # validate category
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")

        delete_image(image_id, category)
        return {"message": f"Image {image_id} deleted successfully from {category}s"}

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.delete("/delete-outfit/{outfit_id}")
async def delete_outfit_endpoint(outfit_id: int):
    """
    delete an outfit by id from the outfits database
    """
    try:
        delete_outfit(outfit_id)
        return {"message": f"Outfit {outfit_id} deleted successfully"}

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-outfit/")
async def upload_outfit(file: UploadFile = File(...)):
    """
    upload a complete outfit image (photo of person wearing full outfit).
    the image will be split into top/bottom halves and embeddings generated for each.
    """
    try:
        outfit_image_data = await file.read()

        # split image and generate embeddings for top and bottom regions
        top_embedding, bottom_embedding = split_and_embed_outfit(outfit_image_data)

        # convert embeddings to bytes for storage
        top_embedding_bytes = pickle.dumps(top_embedding.tolist())
        bottom_embedding_bytes = pickle.dumps(bottom_embedding.tolist())

        # store outfit in database
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
    upload a single clothing item and get a recommendation from your closet.

    workflow:
    1. find an outfit with similar item in the specified category
    2. get the opposite piece from that outfit (if you uploaded top, get outfit's bottom)
    3. search your closet for items similar to that outfit piece
    4. return recommendation from your actual wardrobe

    args:
        file: image of a single clothing item (top or bottom)
        category: 'top' or 'bottom' - which type of clothing item this is

    returns:
        recommendation with item from user's closet and confidence level
    """
    try:
        # validate category
        if category not in ['top', 'bottom']:
            raise HTTPException(status_code=400, detail="Category must be 'top' or 'bottom'")

        # get all outfit embeddings
        outfit_embeddings = get_all_outfit_embeddings()
        if not outfit_embeddings:
            raise HTTPException(status_code=404, detail="No outfits in database to search")

        # generate embedding for the uploaded item
        item_image_data = await file.read()
        item_embedding = get_image_embedding(item_image_data)

        # step 1: find top 5 best matching outfits for uploaded item
        outfit_matches = []

        for outfit_id, top_embedding_bytes, bottom_embedding_bytes in outfit_embeddings:
            # choose which embedding to compare based on category
            if category == 'top':
                outfit_embedding = torch.tensor(pickle.loads(top_embedding_bytes))
                opposite_embedding_bytes = bottom_embedding_bytes
            else:  # category == 'bottom'
                outfit_embedding = torch.tensor(pickle.loads(bottom_embedding_bytes))
                opposite_embedding_bytes = top_embedding_bytes

            # compute similarity
            similarity = F.cosine_similarity(item_embedding, outfit_embedding).item()

            outfit_matches.append({
                "outfit_id": outfit_id,
                "similarity": similarity,
                "opposite_embedding": torch.tensor(pickle.loads(opposite_embedding_bytes))
            })

        # sort by similarity descending and take top 5 outfits
        outfit_matches.sort(key=lambda x: x["similarity"], reverse=True)
        top_5_outfits = outfit_matches[:5]

        if not top_5_outfits:
            raise HTTPException(status_code=404, detail="Could not find matching outfit")

        # step 2: for each of the 5 best outfits, find 3 best matching items from closet
        opposite_category = 'bottom' if category == 'top' else 'top'
        closet_items = get_all_embeddings(opposite_category)

        if not closet_items:
            raise HTTPException(
                status_code=404,
                detail=f"No {opposite_category}s in your closet. Upload some {opposite_category}s first!"
            )

        all_suggestions = []

        for outfit_match in top_5_outfits:
            # for this specific outfit, find best 3 closet items
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

            # sort by similarity and take top 3 for this outfit
            closet_matches.sort(key=lambda x: x["similarity"], reverse=True)
            top_3_closet_items = closet_matches[:3]

            # add these 3 suggestions
            for closet_item in top_3_closet_items:
                all_suggestions.append({
                    "id": closet_item["id"],
                    "filename": closet_item["filename"],
                    "similarity_to_outfit": closet_item["similarity"],
                    "reference_outfit_id": closet_item["outfit_id"]
                })

        # determine confidence level based on best match
        best_similarity = all_suggestions[0]["similarity_to_outfit"] if all_suggestions else 0
        confidence = "high" if best_similarity > 0.6 else "low"
        confidence_message = ""
        if confidence == "low":
            confidence_message = f"These are the closest matches from your closet ({best_similarity*100:.1f}%), but they're not very similar to the styled outfit."

        return {
            "message": f"Based on outfits with similar {category}s, here are suggestions from your closet",
            "outfit_reference_id": top_5_outfits[0]["outfit_id"],  # first/best outfit
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
    get list of all uploaded outfits (metadata only, no image data).
    """
    try:
        return get_all_outfits()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch outfits: {str(e)}")

@app.get("/get-outfit/{outfit_id}")
async def get_outfit(outfit_id: int):
    """
    get a specific outfit image by id and return as binary image.
    """
    try:
        outfit_data = get_outfit_by_id(outfit_id)
        if outfit_data is None:
            raise HTTPException(status_code=404, detail="Outfit not found")

        # outfit_data is tuple: (id, outfit_image_data, top_embedding, bottom_embedding, uploaded_at)
        # return the outfit image binary data (index 1)
        return Response(content=outfit_data[1], media_type="image/jpeg")

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate-random-outfit/")
async def generate_random_outfit():
    """
    generate a random outfit by selecting one random top and one random bottom
    """
    try:
        # get all tops and bottoms
        tops = get_all_images('top')
        bottoms = get_all_images('bottom')

        # handle edge cases
        if not tops and not bottoms:
            raise HTTPException(status_code=404, detail="No tops or bottoms available in the database")

        if not tops:
            raise HTTPException(status_code=404, detail="No tops available in the database")

        if not bottoms:
            raise HTTPException(status_code=404, detail="No bottoms available in the database")

        # randomly select one top and one bottom
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

@app.put("/rename-image/{category}/{image_id}")
async def rename_image_endpoint(category: str, image_id: int, payload: dict):
    """
    rename an image in the specified category

    args:
        category: 'top', 'bottom', or 'outfit'
        image_id: id of the image to rename
        payload: json with 'new_filename' field

    returns:
        success message with new filename
    """
    try:
        # validate category
        if category not in ['top', 'bottom', 'outfit']:
            raise HTTPException(status_code=400, detail="Category must be 'top', 'bottom', or 'outfit'")

        # get new filename from payload
        new_filename = payload.get('new_filename')
        if not new_filename:
            raise HTTPException(status_code=400, detail="new_filename is required")

        # outfits don't have filenames, so reject rename requests for outfits
        if category == 'outfit':
            raise HTTPException(status_code=400, detail="Outfits cannot be renamed (they don't have filenames)")

        # rename the image
        rename_image(image_id, new_filename, category)

        return {
            "message": f"Image renamed successfully",
            "id": image_id,
            "new_filename": new_filename,
            "category": category
        }

    except ValueError as ve:
        raise HTTPException(status_code=409, detail=str(ve))
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))