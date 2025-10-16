from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import clip
from io import BytesIO

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500", # Example port for live server extension
    "http://127.0.0.1:5500",
    # You can also add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for uploaded images and their embeddings
db = {}

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to generate embedding for a single image
def get_image_embedding(image_data):
    image = preprocess(Image.open(BytesIO(image_data))).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.flatten().tolist()

# Route to handle image upload
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    print("1. Upload route triggered.")
    try:
        image_data = await file.read()
        print(f"2. Read {len(image_data)} bytes from the uploaded file.")

        # In a real app, save the image to a file or cloud storage
        image_id = file.filename

        # Generate the embedding
        embedding = get_image_embedding(image_data)

        # Store in our "database"
        db[image_id] = {
            "embedding": embedding,
            # In a real app, you would also store the image URL/path
            "image_data": image_data
        }

        return {"filename": image_id, "message": "Image uploaded and embedded successfully."}

    except Exception as e:
         # Catch any exceptions and return a clear error message
        return {"error": f"An error occurred during upload: {e}"}

# Route to calculate and return similarity score
@app.post("/compare-images/")
async def compare_images(file: UploadFile = File(...)):
    try:
        if not db:
            return {"error": "No images in database to compare against."}

        # Get the new image's embedding
        image_data = await file.read()
        new_embedding = torch.tensor(get_image_embedding(image_data))

        best_match = None
        highest_score = -1

        # Compare with all existing images in the database
        for image_id, data in db.items():
            db_embedding = torch.tensor(data["embedding"])

            # Calculate cosine similarity
            similarity_score = torch.nn.functional.cosine_similarity(new_embedding.unsqueeze(0), db_embedding.unsqueeze(0))

            if similarity_score > highest_score:
                highest_score = similarity_score.item()
                best_match = image_id

        return {
            "best_match": best_match,
            "similarity_score": highest_score
        }
    except Exception as e:
        # Catch any exceptions and return a clear error message
        return {"error": f"An error occurred during comparison: {e}"}