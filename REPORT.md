# Text-Based Person Search System — Project Report

## What This Project Does

Imagine you are looking for a specific person in a large crowd of photos. Instead of clicking through thousands of images one by one, you just type a sentence like **"man in blue jacket carrying a bag"** — and the system finds the most matching photos in about 2 seconds.

That is what this project builds. It is a search engine, but instead of searching for web pages, it searches for people in images using plain English descriptions.

---

## Why This Is Hard

A normal search engine matches words to words. If you type "blue jacket", it looks for files named "blue_jacket.jpg". That does not work for finding people in surveillance photos — the files have names like `0284001.png` which tell you nothing.

This project solves that by teaching a computer to **understand both pictures and words at the same time**, so it can compare "man in blue jacket" with an actual photo and decide how similar they are. This is called **vision-language understanding**, and it is a hard problem in artificial intelligence research.

---

## The AI Model — TBPS-CLIP

The brain of this system is a model called **TBPS-CLIP**, which stands for *Text-Based Person Search with CLIP*.

CLIP (Contrastive Language-Image Pretraining) was originally built by OpenAI. It was trained on hundreds of millions of image-text pairs from the internet, so it learned that the word "red" connects to the color red in photos, that "jacket" looks like a jacket, and so on.

TBPS-CLIP takes the original CLIP model and fine-tunes it specifically for **person re-identification** — the task of finding the same person across different camera angles and lighting conditions. The fine-tuned checkpoint (`best.pth`) was trained on the CUHK-PEDES dataset, which contains 40,206 person images each described by two human-written sentences.

### How the model works

When you type "woman in red jacket", the model does this:

1. It breaks your text into tokens (small pieces of words)
2. It runs those tokens through a **text encoder** — a deep neural network called a Transformer
3. The output is a list of 512 numbers called a **vector** or **embedding**

This vector is like a coordinate in a 512-dimensional space. The key insight is: **images that match your description will have vectors that point in the same direction**.

When an image is processed:
1. It is resized to 224×224 pixels
2. It is normalized using ImageNet mean and standard deviation values
3. It is run through a **visual encoder** (Vision Transformer, ViT-B/16)
4. The output is also a 512-dimensional vector

To compare a text query against an image, we compute the **cosine similarity** between their two vectors — how much they point in the same direction. A score of 1.0 means perfect match, 0.0 means completely unrelated.

Both text and image vectors are **L2-normalized** before comparison, meaning their length is scaled to exactly 1.0. This makes the cosine similarity calculation simple: just multiply the two vectors together element-by-element and sum the result.

---

## System Architecture

The system is split into four parts that talk to each other:

```
User types query
      ↓
[React Frontend]  ←→  [API Gateway]  ←→  [Cloud Qdrant]
                            ↓
                   [Model Service on Colab GPU]
```

### Part 1 — Model Service (Google Colab, GPU)

This is where the AI runs. Google Colab provides a free NVIDIA GPU (T4), which can process images about 20× faster than a regular CPU.

The model service is a web API built with **FastAPI** and **uvicorn**. It exposes these endpoints:

- `POST /encode/text` — takes a text query, returns a 512-float vector
- `POST /encode/image/path` — takes a file path on the Colab machine, returns a vector
- `POST /encode/image/upload` — takes an uploaded image file, returns a vector
- `POST /encode/batch` — scans a folder of images, encodes them all in batches of 32-64, returns all vectors
- `GET /get-image` — reads an image file from disk and serves it as JPEG/PNG

Because Colab runs in Google's cloud and not on a public IP, we use **ngrok** to create a secure tunnel that gives the service a public URL like `https://xxxx.ngrok-free.app`. This URL changes every time a new Colab session starts.

The model loading process on startup:
1. Set distributed training environment variables (required by TBPS-CLIP even for single GPU)
2. Load the YAML config file
3. Override config paths to point to actual Colab file locations
4. Initialize the `clip_vitb` model architecture with `num_classes=11003`
5. Load the fine-tuned weights from `best.pth`
6. Set model to evaluation mode (no gradient computation)

### Part 2 — Cloud Qdrant (Vector Database)

Storing 214,258 vectors of 512 floats each in RAM would require about **440 MB** just for the numbers — before any indexing structure. At 1 million vectors, that becomes over **2 GB**. A regular Python list or NumPy array cannot search this efficiently.

**Qdrant** is a purpose-built vector database. It stores vectors on disk and builds an **HNSW index** (Hierarchical Navigable Small World graph) that allows searching 200,000+ vectors in milliseconds — much faster than checking every vector one by one.

We use the cloud-hosted version at `cloud.qdrant.io` so the data persists between Colab sessions (Colab's disk is wiped when the session ends). Each dataset gets its own **collection** in Qdrant:

| Collection | Images |
|---|---|
| cuhk_pedes | 38,942 |
| rstpreid | 20,505 |
| last | 125,353 |
| market1501 | 12,936 |
| dukemtmc | 16,522 |
| **Total** | **214,258** |

Each stored point has:
- A unique integer ID
- A 512-float vector (L2-normalized)
- A payload with `image_path` (full path on Colab) and `dataset` name

A keyword index is created on the `dataset` payload field so that filtering by dataset during search is fast.

### Part 3 — API Gateway (Local Machine, FastAPI)

This is a thin Python service that runs on the local laptop using `uvicorn`. It acts as the middleman between the frontend and the two backend services. The frontend never talks to Colab or Qdrant directly — everything goes through here.

Built with **FastAPI** and **httpx** for async HTTP calls.

Endpoints:

**`POST /search`**
1. Receive query text, top-K, and optional dataset filter
2. Call Colab model service `/encode/text` → get 512-float vector
3. Call Qdrant `query_points()` on all collections (or just one if filtered)
4. Sort results by cosine similarity score, return top-K
5. Return `image_path`, `score`, `dataset`, and total `latency_ms`

**`GET /image?path=...`**
- The image files live on the Colab machine, not locally
- The frontend cannot reach Colab directly
- This endpoint proxies the request: frontend → API Gateway → Colab `/get-image` → image bytes → frontend
- Uses `StreamingResponse` so it does not load the full image into memory

**`POST /index/dataset`**
- Calls Colab `/encode/batch` to encode all images in a folder
- Upserts vectors to Qdrant in chunks of 100 to avoid memory issues
- Used for adding new datasets (though for large datasets, running directly in Colab is faster due to ngrok timeout limits)

**`GET /stats`** — returns total image count per collection

**`GET /collections`** — returns list of collection names (used to populate the frontend dropdown)

**`GET /health`** — probes both Colab model service and Qdrant, returns their status

### Part 4 — React Frontend

A single-page web application built with:
- **React 18** + **TypeScript** for the UI logic
- **Vite** as the build tool and dev server
- **Tailwind CSS** for styling
- **react-router-dom** for two-page navigation

**Search Page (`/`)**
- Text input for the query
- Dropdown to filter by dataset (populated dynamically from `/collections`)
- Slider to choose Top-K (1 to 20)
- Image grid showing results with similarity score and dataset label on each card
- Shows result count and latency in milliseconds
- Loading spinner while waiting for results

**Manage Page (`/manage`)**
- Table showing all indexed datasets with image counts from `/stats`
- Form to index a new dataset by folder path and name
- Polls `/stats` every 2 seconds while indexing to show live progress

---

## Data Indexing Pipeline

Getting 214,258 images into Qdrant required a GPU encoding pipeline:

1. **Scan** — `glob.glob()` recursively finds all `.jpg`, `.jpeg`, `.png` files in the dataset folder
2. **Load** — each image is opened with PIL and converted to RGB
3. **Transform** — resize to 224×224, convert to tensor, normalize with ImageNet stats
4. **Batch** — group 64 images into a single tensor `[64, 3, 224, 224]`
5. **Encode** — pass batch through model's visual encoder on GPU → `[64, 512]` feature matrix
6. **Normalize** — divide each 512-vector by its L2 norm so all vectors have length 1.0
7. **Buffer** — accumulate 100 encoded points in a list
8. **Upsert** — send 100 points to Qdrant cloud, clear the buffer, repeat

This pipeline runs entirely on Colab GPU and writes directly to cloud Qdrant. Memory usage stays constant regardless of dataset size because only one batch of 64 images is in GPU memory at any time.

---

## Datasets

Five public person re-identification benchmark datasets were indexed:

| Dataset | Description | Images |
|---|---|---|
| CUHK-PEDES | Street surveillance, Hong Kong. Each person has text descriptions | 38,942 |
| RSTPReid | Shopping mall surveillance, 5 text descriptions per person | 20,505 |
| LAST | Large-scale dataset, diverse outdoor scenes | 125,353 |
| Market-1501 | Multi-camera setup, Tsinghua University campus | 12,936 |
| DukeMTMC-reID | Multi-camera, Duke University campus | 16,522 |

Additional large-scale datasets (MSMT17, full LAST split) were not included because they require institutional access agreements and authorization signatures from a supervising professor. The system architecture supports indexing these datasets when access is granted — no code changes are needed, only running the indexing pipeline.

---

## Search Performance

- **Search latency**: ~1.5–2.5 seconds end-to-end
  - ~1.5s for text encoding via ngrok (Colab round-trip)
  - ~50–200ms for Qdrant vector search across 214k vectors
  - ~100–200ms for image proxying per result

- **Similarity scores**: typically 25–30% for good matches
  - This is normal for cross-modal retrieval — cosine similarity on normalized CLIP vectors rarely exceeds 35% even for near-perfect matches
  - The ranking order (which image is #1 vs #10) is what matters, not the absolute score

- **Search quality**: works best with full descriptive sentences
  - Good: `"young man wearing a blue shirt and black pants walking"`
  - Weaker: `"shoes"`, `"purple"` (single attributes, low recall)

---

## Query Caching

To improve response time for repeated queries, the API Gateway implements an in-memory query cache. When a text query is received, the gateway first checks if it has already encoded this exact query before. If yes, it returns the cached 512-float vector immediately without calling the Colab model service at all.

- **First search** for a query: ~2 seconds (Colab GPU round-trip via ngrok)
- **Same query again**: ~200ms (served from cache, no GPU call)
- Cache holds up to 256 unique queries in RAM
- Queries are normalized (trimmed and lowercased) before caching, so "Man in Blue" and "man in blue" hit the same cache entry

This is a standard technique called **cache warming** — before a demo or presentation, running the expected queries once fills the cache so all subsequent searches feel near-instant. It is especially effective for multi-user scenarios where several people are likely to search similar descriptions.

---

## Reranking Design

A deliberate design decision in this system is the use of a **two-stage retrieval pipeline**. Most simple search systems stop after one stage: query the vector database, get the top results, show them. This project goes further with a second stage called reranking, which was designed and added to improve the quality of results shown to the user.

### Why reranking is needed

When images are indexed, each one is encoded once and the resulting 512-float vector is stored in Qdrant. That vector never changes. When a user types a query, Qdrant compares the query vector against those stored vectors and returns the closest matches.

The problem is that these stored vectors are **static snapshots** — they were computed in batch during indexing without knowing what queries would come later. The stored vector for an image captures its general appearance, but the precise ranking of two similar images depends heavily on the exact phrasing of the query. A one-pass retrieval from Qdrant may surface the right images but put them in the wrong order.

Reranking solves this by doing a **fresh, query-aware pass** over the top candidates.

### How it was designed

A dedicated `POST /rerank` endpoint was added to the model service. The API gateway's `/search` endpoint was then redesigned to coordinate two stages:

**Stage 1 — Fast candidate retrieval (Qdrant)**
- Fetch **3× the requested Top-K** from Qdrant using pre-indexed vectors (e.g. 30 candidates when the user wants 10)
- This is fast (~50ms) because Qdrant's HNSW index avoids scanning every vector

**Stage 2 — Precise reranking (GPU)**
- Send the query text and the 30 candidate image paths to `/rerank` on the model service
- The model re-encodes the text query and re-encodes each candidate image fresh from disk on the GPU
- Computes a new dot-product similarity between the live text vector and each live image vector
- Returns 30 fresh scores — the gateway re-sorts and returns the top-K

The key insight is that re-encoding images at query time allows the model to produce vectors more directly comparable to this specific query, rather than relying on a general-purpose vector computed during bulk indexing.

**Graceful fallback**: if the reranking call fails for any reason, the system automatically falls back to the original Qdrant order. Search always returns results.

### Latency impact

| Stage | Time |
|---|---|
| Qdrant retrieval (30 candidates) | ~50–100ms |
| GPU reranking (30 images) | ~1–2s |
| **Total with reranking** | **~3–5s** |
| Total without reranking | ~2s |

The added 1–2 seconds is the tradeoff for better precision — the images shown at positions #1, #2, #3 are more likely to genuinely match the description.

## Limitations

1. **Colab session timeout**: Google Colab disconnects after ~12 hours of inactivity. The session must be restarted and all cells re-run. However, because a fixed ngrok authtoken is configured, ngrok reassigns the same tunnel URL on reconnect — so the `.env` file and API gateway do not need to be updated.

2. **Image availability**: Images are stored on the Colab machine's temporary disk. If the session restarts, datasets must be unzipped again from Google Drive before images can be served.

3. **Single-word queries**: The model was trained on full sentences. Short or single-word queries produce weaker results.

4. **Dataset size**: 214,258 images is a fraction of the 1 million target. The gap is due to institutional access restrictions on larger datasets, not system limitations. Qdrant's HNSW index handles millions of vectors efficiently.

5. **Reranking latency**: The two-stage search (Qdrant retrieval + GPU reranking) adds 1–2 seconds compared to retrieval-only. For a demo context this is acceptable, but a production system would cache rerank scores or use a lighter reranker.
