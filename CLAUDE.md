# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

"Instant Outfitters" - A CLIP-based image similarity application for matching clothing items. The application uses OpenAI's CLIP (Contrastive Language-Image Pre-Training) model to generate embeddings for uploaded images and compute similarity scores between them.

## Architecture

### Backend (FastAPI)
- **Location**: `clip_workingtrial/app/`
- **Framework**: FastAPI with CORS middleware
- **Main components**:
  - `main.py` - FastAPI application with three endpoints:
    - `POST /upload-image/` - Upload image, generate CLIP embedding, store in DB, compute similarities with existing images
    - `POST /compare-images/` - Upload image and find best match from database
    - `GET /get-images/` - Retrieve list of all uploaded images
  - `database.py` - SQLite database layer with two tables:
    - `images` table - stores filename, image_data (BLOB), embedding (BLOB), and upload timestamp
    - `similarities` table - stores pairwise similarity scores between images
  - Database file: `clip_workingtrial/app/images.db`

### Frontend (HTML/JavaScript)
- **Location**: `clip_workingtrial/frontend/index.html`
- **Type**: Simple HTML with vanilla JavaScript
- **Features**: Upload form for embedding images, comparison form, and list of uploaded images
- **CORS configuration**: Expects backend on `http://127.0.0.1:8000`

### CLIP Model
- **Location**: `clip_workingtrial/CLIP/` (OpenAI CLIP library)
- **Model variant**: ViT-B/32
- **Device**: Auto-detects CUDA/CPU
- **Embedding**: Images are preprocessed, encoded, and normalized (L2 normalization)
- **Similarity**: Computed using cosine similarity between embeddings

## Development Setup

### Environment Setup
```bash
cd clip_workingtrial
source venv/bin/activate  # Activate virtual environment
```

### Installing Dependencies
The CLIP library requirements are in `clip_workingtrial/CLIP/requirements.txt`:
```bash
pip install ftfy packaging regex tqdm torch torchvision
```

Additional FastAPI dependencies (not in requirements.txt, infer from imports):
```bash
pip install fastapi uvicorn python-multipart pillow
```

### Running the Application

1. **Start the backend server**:
```bash
cd clip_workingtrial
source venv/bin/activate
uvicorn app.main:app --reload
```
Backend will run on `http://127.0.0.1:8000`

2. **Start the frontend**:
Open `clip_workingtrial/frontend/index.html` with a live server (e.g., VS Code Live Server extension):
- Right-click in the editor
- Select "Open with Live Server"

### Testing CLIP Locally
The `clip_workingtrial/clip_test.py` script demonstrates basic CLIP usage:
- Loads images from `clip_workingtrial/images/` directory
- Generates embeddings for each image
- Computes similarity between first two images
```bash
cd clip_workingtrial
python clip_test.py
```

## Key Technical Details

### Embedding Storage
- Embeddings are converted to lists, serialized with `pickle.dumps()`, and stored as BLOBs in SQLite
- When retrieved, they're deserialized with `pickle.loads()` and converted back to tensors

### Similarity Computation
- Uses PyTorch's `F.cosine_similarity()` for comparing embeddings
- Returns values between -1 and 1 (1 = identical, -1 = completely opposite)
- When uploading an image, similarities are computed against all existing images in the database

### Database Initialization
- Database is auto-initialized on FastAPI startup via `init_db()` in `main.py`
- Creates tables if they don't exist

## Project Structure
```
FA25-Group19/
├── clip_workingtrial/
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── database.py       # SQLite database layer
│   │   └── images.db         # SQLite database file
│   ├── frontend/
│   │   └── index.html        # Simple web UI
│   ├── CLIP/                 # OpenAI CLIP library
│   ├── images/               # Test images directory
│   ├── venv/                 # Python virtual environment
│   ├── clip_test.py          # Standalone CLIP test script
│   └── how_to_start.txt      # Quick start guide
├── Docs/
│   ├── PLAN.md
│   └── RUN.md
└── README.md
```

## Git Information
- **Current branch**: `zoya`
- **Main branch**: `master`
- Working directory has uncommitted changes to database and compiled Python files
