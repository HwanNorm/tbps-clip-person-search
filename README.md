# Person Search — TBPS-CLIP

Text-based person search powered by TBPS-CLIP. Type a description like "man in white shirt" and get matching person images from the database.

## Architecture

```
Colab (GPU)          Cloud Qdrant         Your Laptop
-----------          ------------         -----------
Model Service   -->  Vector Storage  <--  API Gateway (port 8000)
(encode text/        (38,942 images        + Frontend  (port 3000)
 image via ngrok)     indexed)
```

## Prerequisites

- Google Colab (GPU runtime)
- Cloud Qdrant account (free tier works)
- Python 3.x + Node 20 on your laptop
- Files on Google Drive:
  - `best.pth` — finetuned TBPS-CLIP checkpoint
  - `simplified-ViT-B-16.pth` — base CLIP backbone
  - `s.config.yaml` — model config
  - `CUHK-PEDES.zip` — image dataset
  - `annotation.zip` — dataset annotations

---

## Step 1 — Start the Model Service on Colab

1. Open `colab_gpu_2.py` in Google Colab (File → Upload, or use the saved notebook)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Paste your ngrok auth token in Cell 7 (`PASTE_YOUR_NGROK_TOKEN_HERE`)
4. Run all cells top-to-bottom — takes ~5 minutes
5. Copy the ngrok URL printed at the end, e.g. `https://xxxx.ngrok-free.app`

---

## Step 2 — Configure `.env`

Edit `person-search/.env`:

```
MODEL_SERVICE_URL=https://xxxx.ngrok-free.app   # from Colab output
NGROK_AUTH_TOKEN=your_ngrok_token_here
QDRANT_URL=https://xxxx.aws.cloud.qdrant.io      # your cloud Qdrant endpoint
QDRANT_API_KEY=eyJhbGc...                        # your cloud Qdrant API key
```

---

## Step 3 — Run the API Gateway

```bash
cd person-search/api-gateway
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Verify it's working:
```
http://localhost:8000/health
```

Expected: `{"status":"ok","model_service":"ok","qdrant":"ok"}`

API docs available at: `http://localhost:8000/docs`

---

## Step 4 — Run the Frontend

```bash
cd person-search/frontend
npm install
npm run dev
```

Open: `http://localhost:3000`

---

## Step 5 — Index a Dataset (first time only)

CUHK-PEDES is already indexed (38,942 images). To index a new dataset, go to `http://localhost:3000/manage` and fill in:

- **Dataset name**: e.g. `msmt17`
- **Folder path**: path on the Colab machine, e.g. `/content/drive/MyDrive/MSMT17/imgs`

Or run it directly in a Colab cell (faster, avoids ngrok timeout):

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from PIL import Image
from torchvision import transforms
import glob, os, torch

QDRANT_URL = "..."
QDRANT_API_KEY = "..."
DATASET_NAME = "msmt17"
FOLDER_PATH = "/content/MSMT17/imgs"

# ... (see indexing cell in colab_gpu_2.py)
```

---

## Project Structure

```
person-search/
├── .env                        # secrets — never commit this
├── README.md
├── colab_gpu_2.py              # Colab notebook: model service
├── api-gateway/
│   ├── main.py                 # FastAPI gateway
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── pages/
    │   │   ├── Search.tsx      # main search page
    │   │   └── Manage.tsx      # dataset manager
    │   ├── api.ts              # API calls
    │   └── main.tsx
    └── package.json
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check model service + Qdrant status |
| POST | `/search` | Search by text query |
| GET | `/image?path=...` | Serve image file |
| GET | `/collections` | List indexed datasets |
| GET | `/stats` | Image counts per dataset |
| POST | `/index/dataset` | Index a new dataset |

---

## Notes

- ngrok URL changes every Colab session — update `.env` and restart the API gateway each time
- Colab free tier disconnects after ~12 hours — re-run Cells 5-7 to restart the model service (no need to re-index)
- Search scores around 25-30% are normal for this dataset — ranking order is what matters
