# -*- coding: utf-8 -*-
"""
TBPS-CLIP Model Service — Colab GPU
Run cells top-to-bottom once per new session.
"""

# ============================================================
# CELL 1 — Install dependencies
# ============================================================
!pip install fastapi uvicorn pyngrok python-multipart ftfy Pillow qdrant-client
!pip install git+https://github.com/openai/CLIP.git
!git clone https://github.com/Flame-Chasers/TBPS-CLIP.git
!pip install -r TBPS-CLIP/requirements.txt

import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available())

# ============================================================
# CELL 2 — NLTK downloads
# ============================================================
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ============================================================
# CELL 3 — Mount Drive & copy model files
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/TBPS-CLIP/ckpts/s.baseline/CUHK-PEDES', exist_ok=True)

!cp /content/drive/MyDrive/best.pth /content/TBPS-CLIP/ckpts/s.baseline/CUHK-PEDES/best.pth
!cp /content/drive/MyDrive/simplified-ViT-B-16.pth /content/TBPS-CLIP/simplified-ViT-B-16.pth
!cp /content/drive/MyDrive/s.config.yaml /content/TBPS-CLIP/config/s.config.yaml
!unzip -q /content/drive/MyDrive/CUHK-PEDES.zip -d /content/CUHK-PEDES/
!unzip -q /content/drive/MyDrive/annotation.zip -d /content/annotations/
!unzip -q /content/drive/MyDrive/RSTPReid.zip -d /content/RSTPReid/
!unzip -q /content/drive/MyDrive/Market-1501-v15.09.15.zip -d /content/Market-1501/
!unzip -q /content/drive/MyDrive/DukeMTMC-reID.zip -d /content/DukeMTMC/
!unzip -q /content/drive/MyDrive/last.zip -d /content/LAST/
print("✅ All files ready!")

# ============================================================
# CELL 4 — Fix build.py (required by TBPS-CLIP)
# ============================================================
build_py = r'''
import os
import torch
import numpy as np
import math
import torch.nn.functional as F


def resize_pos_embed(posemb, posemb_new, hight, width):
    posemb = posemb.unsqueeze(0)
    posemb_new = posemb_new.unsqueeze(0)
    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb.squeeze(0)


def interpolate_text(pos_embed_checkpoint, target_dim=77):
    if pos_embed_checkpoint.size(0) == target_dim:
        return pos_embed_checkpoint
    start_token = pos_embed_checkpoint[:1, :]
    end_token = pos_embed_checkpoint[-1:, :]
    pos_tokens = pos_embed_checkpoint[1:-1, :].unsqueeze(0).permute(0, 2, 1)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=target_dim - 2, mode='linear')
    pos_tokens = pos_tokens.squeeze(0).t()
    pos_tokens = torch.cat([start_token, pos_tokens, end_token], dim=0)
    return pos_tokens


def load_checkpoint(model, config):
    if config.model.ckpt_type == 'original_clip':
        with open(config.model.checkpoint, 'rb') as opened_file:
            model_tmp = torch.jit.load(opened_file, map_location="cpu")
            state = model_tmp.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state:
                del state[key]
        new_state = {}
        for name, params in state.items():
            if name == 'visual.positional_embedding' and params.shape != model.visual.positional_embedding.shape:
                params = resize_pos_embed(params, model.visual.positional_embedding, model.visual.num_y, model.visual.num_x)
            if name == 'positional_embedding':
                new_state['encode_text.' + name] = interpolate_text(params, config.experiment.text_length)
            elif name.startswith('transformer') or name in ['positional_embedding', 'token_embedding.weight',
                                                            'ln_final.weight', 'ln_final.bias', 'text_projection']:
                new_state['encode_text.' + name] = params
            else:
                new_state[name] = params
    elif config.model.ckpt_type == 'saved':
        ckpt = torch.load(config.model.saved_path, map_location='cpu', weights_only=False)
        new_state = ckpt['model']
    else:
        raise KeyError

    load_result = model.load_state_dict(new_state, strict=False)
    return model, load_result


def build_optimizer(config, model):
    params = []
    schedule_config = config.schedule
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        weight_decay = schedule_config.weight_decay
        ratio = 1.
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            weight_decay = 0.
        if "cross" in n or "classifier" in n or "mlm_head" in n:
            ratio = ratio * schedule_config.ratio_factor
        params += [{"params": [p], "weight_decay": weight_decay, "ratio": ratio}]
    optimizer = torch.optim.AdamW(params, lr=schedule_config.lr, betas=schedule_config.betas,
                                  eps=schedule_config.eps, weight_decay=schedule_config.weight_decay)
    return optimizer
'''

with open('/content/TBPS-CLIP/misc/build.py', 'w') as f:
    f.write(build_py)
print("✅ build.py fixed!")

# ============================================================
# CELL 5 — Load model in notebook scope (used for indexing)
# ============================================================
import os, sys, torch
import torch.distributed as dist

if dist.is_initialized():
    dist.destroy_process_group()

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

sys.path.insert(0, '/content/TBPS-CLIP')
os.chdir('/content/TBPS-CLIP')

import importlib
import misc.build
importlib.reload(misc.build)

from misc.utils import parse_config, init_distributed_mode, set_seed
from misc.build import load_checkpoint
from model.tbps_model import clip_vitb
from misc.data import build_pedes_data
import clip

config = parse_config('config/s.config.yaml')
config.model.saved_path = '/content/TBPS-CLIP/ckpts/s.baseline/CUHK-PEDES/best.pth'
config.model.checkpoint  = '/content/TBPS-CLIP/simplified-ViT-B-16.pth'
config.data.ann_root     = '/content/annotations'
config.data.image_root   = '/content/CUHK-PEDES/CUHK-PEDES/imgs'
config.device            = 'cuda' if torch.cuda.is_available() else 'cpu'

init_distributed_mode(config)
set_seed(config)

dataloader = build_pedes_data(config)
num_classes = len(dataloader['train_loader'].dataset.person2text)

model = clip_vitb(config, num_classes)
model.to(config.device)
model, _ = load_checkpoint(model, config)
model.eval()
print(f"✅ Model loaded on {config.device} | num_classes={num_classes}")

# ============================================================
# CELL 6 — Write model_service.py to disk
# ============================================================
model_service_code = '''
import os, sys, glob, io, mimetypes
import torch
import clip
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms
from pathlib import Path

sys.path.insert(0, "/content/TBPS-CLIP")
os.chdir("/content/TBPS-CLIP")

from misc.utils import parse_config, init_distributed_mode, set_seed
from misc.build import load_checkpoint
from model.tbps_model import clip_vitb

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Model loads itself on startup so uvicorn subprocess is self-contained
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = parse_config("/content/TBPS-CLIP/config/s.config.yaml")
config.model.saved_path = "/content/TBPS-CLIP/ckpts/s.baseline/CUHK-PEDES/best.pth"
config.device = device

model = clip_vitb(config, 11003)
model.to(device)
model, _ = load_checkpoint(model, config)
model.eval()
print(f"✅ Model loaded inside service on {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TextRequest(BaseModel):
    text: str

class ImagePathRequest(BaseModel):
    image_path: str

class BatchRequest(BaseModel):
    folder_path: str
    batch_size: int = 32

class RerankRequest(BaseModel):
    text: str
    image_paths: list


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.get("/get-image")
def get_image(path: str = Query(...)):
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")
    mime, _ = mimetypes.guess_type(str(p))
    return FileResponse(str(p), media_type=mime or "image/jpeg")


@app.post("/encode/text")
def encode_text(req: TextRequest):
    tokens = clip.tokenize([req.text], context_length=77).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return {"vector": feat.cpu().tolist()[0]}


@app.post("/encode/image/path")
def encode_image_path(req: ImagePathRequest):
    img = Image.open(req.image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return {"vector": feat.cpu().tolist()[0]}


@app.post("/encode/image/upload")
async def encode_image_upload(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return {"vector": feat.cpu().tolist()[0]}


@app.post("/encode/batch")
def encode_batch(req: BatchRequest):
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(req.folder_path, "**", ext), recursive=True))
    image_paths = sorted(set(image_paths))
    results = []
    for i in range(0, len(image_paths), req.batch_size):
        batch_paths = image_paths[i : i + req.batch_size]
        tensors, valid_paths = [], []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(transform(img))
                valid_paths.append(path)
            except Exception:
                continue
        if not tensors:
            continue
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        for path, feat in zip(valid_paths, feats):
            results.append({"image_path": path, "vector": feat.cpu().tolist()})
        del batch_tensor, feats
    return {"total": len(results), "items": results}


@app.post("/rerank")
def rerank(req: RerankRequest):
    """Re-encode candidate images fresh from disk and score against text query."""
    tokens = clip.tokenize([req.text], context_length=77).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    scores = []
    for path in req.image_paths:
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = model.encode_image(tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            scores.append((text_feat @ img_feat.T).item())
        except Exception:
            scores.append(0.0)

    return {"scores": scores}
'''

with open('/content/model_service.py', 'w') as f:
    f.write(model_service_code)
print("✅ model_service.py written!")

# ============================================================
# CELL 7 — Start uvicorn + expose via ngrok
# ============================================================
import subprocess, time, requests
from pyngrok import ngrok

# Kill any leftover uvicorn from a previous run
subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)
time.sleep(2)

proc = subprocess.Popen(
    ["uvicorn", "model_service:app", "--host", "0.0.0.0", "--port", "8001"],
    cwd="/content",
)
time.sleep(20)  # wait for model to load inside the subprocess

# Verify locally before exposing
r = requests.get("http://localhost:8001/health")
print("Local health check:", r.json())

# Expose via ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")  # https://dashboard.ngrok.com/authtokens
public_url = ngrok.connect(8001)
print(f"\n✅ Model Service live at: {public_url}")
print("Paste this into your local .env as MODEL_SERVICE_URL")

# ============================================================
# CELL 8 — Index a dataset into Qdrant
#   Change DATASET_NAME and FOLDER_PATH then run this cell.
#   It is safe to re-run — it resumes from where it left off.
# ============================================================
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, PayloadSchemaType
from PIL import Image
from torchvision import transforms
import glob, os, torch

QDRANT_URL     = "YOUR_QDRANT_URL"      # e.g. https://xxxx.cloud.qdrant.io
QDRANT_API_KEY = "YOUR_QDRANT_API_KEY"

# ---- CHANGE THESE TWO LINES ----
DATASET_NAME = "cuhk_pedes"
FOLDER_PATH  = "/content/CUHK-PEDES/CUHK-PEDES/imgs"
# ---- dataset paths reference ----
# cuhk_pedes  → /content/CUHK-PEDES/CUHK-PEDES/imgs
# rstpreid    → /content/RSTPReid/imgs
# market1501  → /content/Market-1501/bounding_box_train
# dukemtmc    → /content/DukeMTMC/bounding_box_train
# last        → /content/LAST/last/test/gallery

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

existing = {c.name for c in client.get_collections().collections}
if DATASET_NAME not in existing:
    client.create_collection(
        collection_name=DATASET_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=DATASET_NAME,
        field_name="dataset",
        field_schema=PayloadSchemaType.KEYWORD,
    )

index_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(FOLDER_PATH, "**", ext), recursive=True))
image_paths = sorted(set(image_paths))
print(f"Found {len(image_paths)} images")

BATCH_SIZE = 64
CHUNK = 100
point_id = client.get_collection(DATASET_NAME).points_count  # resume-safe
already_indexed = point_id
buffer = []

for i in range(already_indexed, len(image_paths), BATCH_SIZE):
    batch_paths = image_paths[i : i + BATCH_SIZE]
    tensors, valid_paths = [], []
    for path in batch_paths:
        try:
            img = Image.open(path).convert("RGB")
            tensors.append(index_transform(img))
            valid_paths.append(path)
        except Exception:
            continue

    if not tensors:
        continue

    batch_tensor = torch.stack(tensors).to(config.device)
    with torch.no_grad():
        feats = model.encode_image(batch_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    for path, feat in zip(valid_paths, feats):
        buffer.append(PointStruct(
            id=point_id,
            vector=feat.cpu().tolist(),
            payload={"image_path": path, "dataset": DATASET_NAME},
        ))
        point_id += 1

    if len(buffer) >= CHUNK:
        client.upsert(collection_name=DATASET_NAME, points=buffer)
        buffer = []
        if point_id % 5000 == 0:
            print(f"  Indexed {point_id}/{len(image_paths)}")

if buffer:
    client.upsert(collection_name=DATASET_NAME, points=buffer)

print(f"✅ Done! Total indexed: {point_id}")
