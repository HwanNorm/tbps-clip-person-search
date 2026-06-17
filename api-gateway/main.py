import os
import time
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, FieldCondition, MatchValue, PointStruct, VectorParams

load_dotenv()

MODEL_SERVICE_URL = os.environ["MODEL_SERVICE_URL"].rstrip("/")
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

app = FastAPI(title="Person Search API Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


@app.get("/health")
async def health():
    # Probe model service
    model_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{MODEL_SERVICE_URL}/health")
            model_ok = r.status_code == 200
    except Exception:
        pass

    # Probe Qdrant
    qdrant_ok = False
    try:
        qdrant.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "model_service": "ok" if model_ok else "unreachable",
        "qdrant": "ok" if qdrant_ok else "unreachable",
    }


@app.get("/image")
async def serve_image(path: str = Query(...)):
    """Proxy image from Colab machine through the model service."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{MODEL_SERVICE_URL}/get-image",
            params={"path": path},
        )
    if r.status_code == 404:
        raise HTTPException(status_code=404, detail="Image not found")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Could not fetch image")
    return StreamingResponse(
        iter([r.content]),
        media_type=r.headers.get("content-type", "image/jpeg"),
    )


@app.get("/collections")
def list_collections():
    return [c.name for c in qdrant.get_collections().collections]


@app.get("/stats")
def stats():
    collections = qdrant.get_collections().collections
    counts = {c.name: qdrant.get_collection(c.name).points_count for c in collections}
    return {"total": sum(counts.values()), "collections": counts}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    dataset: str = "all"


_vector_cache: dict[str, list] = {}


async def _encode_text(text: str) -> list:
    if text in _vector_cache:
        return _vector_cache[text]
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{MODEL_SERVICE_URL}/encode/text",
            json={"text": text},
        )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Model service error: {r.text}")
    vector = r.json()["vector"]
    if len(_vector_cache) < 256:
        _vector_cache[text] = vector
    return vector


@app.post("/search")
async def search(req: SearchRequest):
    t0 = time.monotonic()

    query_text = req.query.strip().lower()

    t_encode_start = time.monotonic()
    vector = await _encode_text(query_text)
    encode_ms = round((time.monotonic() - t_encode_start) * 1000)

    # Fetch a larger candidate pool for reranking (3× top_k, minimum 30)
    candidate_k = max(req.top_k * 3, 30)

    # Build filter if targeting a specific dataset
    q_filter = None
    if req.dataset != "all":
        q_filter = Filter(
            must=[FieldCondition(key="dataset", match=MatchValue(value=req.dataset))]
        )

    def _query(col: str, limit: int):
        return qdrant.query_points(
            collection_name=col,
            query=vector,
            limit=limit,
            with_payload=True,
            query_filter=q_filter,
        ).points

    t_retrieve_start = time.monotonic()
    if req.dataset == "all":
        collections = [c.name for c in qdrant.get_collections().collections]
        hits = []
        for col in collections:
            hits.extend(_query(col, candidate_k))
        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:candidate_k]
    else:
        hits = _query(req.dataset, candidate_k)
    retrieve_ms = round((time.monotonic() - t_retrieve_start) * 1000)

    # Rerank: send candidate image paths to model service for precise re-scoring
    rerank_ms = 0
    image_paths = [h.payload.get("image_path", "") for h in hits]
    try:
        t_rerank_start = time.monotonic()
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{MODEL_SERVICE_URL}/rerank",
                json={"text": query_text, "image_paths": image_paths},
            )
        if r.status_code == 200:
            rerank_scores = r.json()["scores"]
            for hit, rscore in zip(hits, rerank_scores):
                hit._rerank_score = rscore
            hits.sort(key=lambda h: getattr(h, "_rerank_score", h.score), reverse=True)
        rerank_ms = round((time.monotonic() - t_rerank_start) * 1000)
    except Exception:
        pass

    hits = hits[: req.top_k]

    results = [
        {
            "image_path": h.payload.get("image_path", ""),
            "score": round(getattr(h, "_rerank_score", h.score), 4),
            "dataset": h.payload.get("dataset", ""),
        }
        for h in hits
    ]

    return {
        "results": results,
        "latency_ms": round((time.monotonic() - t0) * 1000),
        "timing": {
            "encode_ms": encode_ms,
            "retrieve_ms": retrieve_ms,
            "rerank_ms": rerank_ms,
        },
    }


class IndexRequest(BaseModel):
    folder_path: str
    dataset_name: str
    batch_size: int = 32


def _ensure_collection(name: str) -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if name not in existing:
        qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )


def _collection_count(name: str) -> int:
    try:
        return qdrant.get_collection(name).points_count
    except Exception:
        return 0


@app.post("/index/dataset")
async def index_dataset(req: IndexRequest):
    _ensure_collection(req.dataset_name)

    # Call Colab to encode all images — may take several minutes for large datasets
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(
            f"{MODEL_SERVICE_URL}/encode/batch",
            json={"folder_path": req.folder_path, "batch_size": req.batch_size},
        )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Model service error: {r.text}")

    items = r.json().get("items", [])
    base_id = _collection_count(req.dataset_name)

    # Upsert to Qdrant in chunks of 100 — never hold all vectors in memory
    CHUNK = 100
    for i in range(0, len(items), CHUNK):
        chunk = items[i : i + CHUNK]
        points = [
            PointStruct(
                id=base_id + i + j,
                vector=item["vector"],
                payload={"image_path": item["image_path"], "dataset": req.dataset_name},
            )
            for j, item in enumerate(chunk)
        ]
        qdrant.upsert(collection_name=req.dataset_name, points=points)

    return {"indexed": len(items), "dataset": req.dataset_name, "status": "done"}
