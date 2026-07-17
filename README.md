# PlantScan — Plant Disease Classifier

VGG16-based plant disease classifier with FastAPI backend + HTML frontend, deployable on Render.

---

## Step 1 — Fix your notebook before exporting

Add these two cells at the end of your Colab notebook and run them:

### Cell A — Save model in new format + class names
```python
import json

# Save class names (required for prediction labels)
class_names = train_set.class_names
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print(f"Saved {len(class_names)} class names.")

# Save model in .keras format (preferred over .h5)
model.save("vgg16_plant_disease_model.keras")
print("Model saved.")
```

### Cell B — Download both files to your PC
```python
from google.colab import files
files.download("vgg16_plant_disease_model.keras")
files.download("class_names.json")
```

---

## Project Structure

```
plant-disease-app/
├── backend/
│   └── app.py                   # FastAPI app
├── frontend/
│   └── index.html               # Frontend UI
├── vgg16_plant_disease_model.keras   # ← put downloaded model here
├── class_names.json                  # ← put downloaded class names here
├── requirements.txt
└── render.yaml
```

---

## Step 2 — Run locally to test

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (from project root)
uvicorn backend.app:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

---

## Step 3 — Deploy on Render

1. Push this entire folder to a **GitHub repo** (include the `.keras` and `.json` files, or use Git LFS for large model files).

2. Go to https://render.com → **New** → **Web Service**

3. Connect your GitHub repo.

4. Render settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
   - **Python version:** 3.11

5. Click **Deploy** — Render will build and serve your app.

> **Note on model size:** VGG16 `.keras` files are typically 500MB–1GB.
> GitHub has a 100MB file limit, so use **Git LFS** for the model file:
> ```bash
> git lfs install
> git lfs track "*.keras"
> git add .gitattributes
> git add vgg16_plant_disease_model.keras
> git commit -m "Add model via LFS"
> ```
> Alternatively, host the model on Google Drive and download it at startup (ask if you need this approach).

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Check if model is loaded |
| POST | `/predict` | Upload image → get top-3 disease predictions |
| GET | `/` | Serves the frontend |

### Example predict response
```json
{
  "predictions": [
    { "plant": "Apple", "disease": "Apple Scab", "confidence": 94.32 },
    { "plant": "Apple", "disease": "Black Rot",  "confidence":  3.21 },
    { "plant": "Apple", "disease": "Cedar Rust", "confidence":  1.87 }
  ]
}
```
"# PLANT-DISEASE-DETECTOR" 
