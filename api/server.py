"""Gemini ベクトルを用いたカテゴリ検索 API."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import requests
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, RootModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "final_categories_embeddings.jsonl"
DEFAULT_CATEGORIES_PATH = PROJECT_ROOT / "final_categories.json"
DEFAULT_MODEL = "models/text-embedding-004"
DEFAULT_API_VERSION = "v1beta"


class SearchRequest(BaseModel):
    """検索リクエスト。"""

    query: str | None = Field(None, description="Gemini で埋め込みを生成するテキスト")
    embedding: List[float] | None = Field(
        None,
        description="既に生成済みの埋め込み。query を指定しない場合に利用",
    )
    top_k: int = Field(5, ge=1, le=100)
    attributes: List[str] | None = Field(
        None, description="検索対象とする属性。未指定なら全属性"
    )

    model_config = {"extra": "forbid"}

    def embedding_source(self) -> str:
        return "embedding" if self.embedding is not None else "query"


class CategoryInfo(BaseModel):
    category_id: str
    top_level: str | None = None
    domain: str | None = None
    kana_group: str | None = None
    entity: str | None = None
    category_group: str | None = None
    subcategory: str | None = None
    target: str | None = None
    item: str | None = None
    size: str | None = None
    extra: List[str] | None = None
    raw_path: str | None = None


class SearchResult(BaseModel):
    category_id: str
    attribute: str
    score: float
    text: str
    category: CategoryInfo


class SearchMetadata(BaseModel):
    top_k: int
    attributes: List[str]
    embedding_source: str
    result_count: int
    attribute_counts: Dict[str, int]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    meta: SearchMetadata


@dataclass
class Settings:
    embeddings_path: Path
    categories_path: Path
    gemini_api_key: str | None
    gemini_model: str
    gemini_api_version: str
    gemini_timeout: float


def _read_env_path(key: str, default: Path) -> Path:
    value = os.environ.get(key)
    if not value:
        return default
    return Path(value).expanduser().resolve()


def load_settings() -> Settings:
    return Settings(
        embeddings_path=_read_env_path("EMBEDDINGS_PATH", DEFAULT_EMBEDDINGS_PATH),
        categories_path=_read_env_path("CATEGORIES_PATH", DEFAULT_CATEGORIES_PATH),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        gemini_model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        gemini_api_version=os.environ.get("GEMINI_API_VERSION", DEFAULT_API_VERSION),
        gemini_timeout=float(os.environ.get("GEMINI_TIMEOUT", "60")),
    )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class EmbeddingIndex:
    """JSON Lines の埋め込みデータをロードして検索できるようにする。"""

    def __init__(self, embeddings_path: Path, categories_path: Path):
        self.embeddings_path = embeddings_path
        self.categories_path = categories_path
        self.attribute_to_matrix: Dict[str, np.ndarray] = {}
        self.attribute_to_records: Dict[str, List[dict]] = {}
        self.category_map: Dict[str, dict] = {}
        self.dimension: int | None = None
        self._load()

    def _load(self) -> None:
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"embeddings file not found: {self.embeddings_path}")
        if not self.categories_path.exists():
            raise FileNotFoundError(f"categories file not found: {self.categories_path}")

        attribute_vectors: Dict[str, List[np.ndarray]] = defaultdict(list)
        attribute_records: Dict[str, List[dict]] = defaultdict(list)

        with self.embeddings_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                attribute = data.get("attribute")
                embedding = data.get("embedding")
                if not attribute or not embedding:
                    continue
                vector = np.asarray(embedding, dtype=np.float32)
                if vector.size == 0:
                    continue
                if self.dimension is None:
                    self.dimension = vector.size
                elif vector.size != self.dimension:
                    raise ValueError(
                        "embedding dimension mismatch: "
                        f"expected {self.dimension}, got {vector.size}"
                    )
                attribute_vectors[attribute].append(vector)
                attribute_records[attribute].append(
                    {
                        "category_id": data.get("category_id"),
                        "text": data.get("text", ""),
                    }
                )

        if self.dimension is None:
            raise ValueError("no embeddings loaded")

        for attribute, vectors in attribute_vectors.items():
            matrix = np.vstack(vectors).astype(np.float32, copy=False)
            self.attribute_to_matrix[attribute] = _normalize_rows(matrix)
            self.attribute_to_records[attribute] = attribute_records[attribute]

        with self.categories_path.open("r", encoding="utf-8") as f:
            categories = json.load(f)
        self.category_map = {entry.get("category_id"): entry for entry in categories if entry}

    @property
    def available_attributes(self) -> List[str]:
        return sorted(self.attribute_to_matrix.keys())

    def search(self, query_vector: np.ndarray, attributes: Iterable[str], top_k: int) -> List[dict]:
        if self.dimension is None:
            raise RuntimeError("embedding index is not initialized")
        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"embedding dimension mismatch: expected {self.dimension}, got {query_vector.shape[0]}"
            )

        normalized_query = query_vector.astype(np.float32, copy=False)
        norm = np.linalg.norm(normalized_query)
        if norm == 0.0:
            raise ValueError("query embedding norm is zero")
        normalized_query = normalized_query / norm

        results: List[dict] = []
        for attribute in attributes:
            matrix = self.attribute_to_matrix.get(attribute)
            records = self.attribute_to_records.get(attribute)
            if matrix is None or records is None or not len(records):
                continue

            scores = matrix @ normalized_query
            k = min(top_k, scores.shape[0])
            if k == 0:
                continue
            indices = np.argpartition(scores, -k)[-k:]
            ordered = indices[np.argsort(scores[indices])[::-1]]

            for idx in ordered:
                record = records[idx]
                results.append(
                    {
                        "category_id": record.get("category_id", ""),
                        "attribute": attribute,
                        "score": float(scores[idx]),
                        "text": record.get("text", ""),
                    }
                )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def get_category(self, category_id: str) -> CategoryInfo:
        data = self.category_map.get(category_id, {"category_id": category_id})
        return CategoryInfo.model_validate(data)


class GeminiEmbeddingClient:
    def __init__(self, *, api_key: str, api_version: str, model: str, timeout: float = 60.0):
        self.api_key = api_key
        self.api_version = api_version
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> np.ndarray:
        if not text:
            raise ValueError("text must not be empty")

        url = f"https://generativelanguage.googleapis.com/{self.api_version}/{self.model}:batchEmbedContents"
        payload = {
            "requests": [
                {
                    "model": self.model,
                    "content": {"parts": [{"text": text}]},
                }
            ]
        }

        response = requests.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini API error {response.status_code}: {response.text}"
            )

        data = response.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(f"unexpected response payload: {data}")

        vector = embeddings[0].get("values")
        if not vector:
            raise RuntimeError(f"embedding not found in response: {data}")
        return np.asarray(vector, dtype=np.float32)


def _setup_cors(app: FastAPI) -> None:
    env_value = os.environ.get("CORS_ALLOW_ORIGINS", "")
    if env_value.strip() == "*":
        allow_origins = ["*"]
    elif env_value.strip():
        allow_origins = [origin.strip() for origin in env_value.split(",") if origin.strip()]
    else:
        allow_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


app = FastAPI(title="Categorizer API", version="0.1.0")
_setup_cors(app)


@lru_cache()
def get_settings() -> Settings:
    return load_settings()


@lru_cache()
def get_index() -> EmbeddingIndex:
    settings = get_settings()
    return EmbeddingIndex(settings.embeddings_path, settings.categories_path)


def get_gemini_client() -> GeminiEmbeddingClient | None:
    settings = get_settings()
    if not settings.gemini_api_key:
        return None
    return GeminiEmbeddingClient(
        api_key=settings.gemini_api_key,
        api_version=settings.gemini_api_version,
        model=settings.gemini_model,
        timeout=settings.gemini_timeout,
    )


@app.on_event("startup")
def ensure_index_loaded() -> None:
    # キャッシュを温めるため呼び出す
    get_index()


@app.get("/healthz")
def healthcheck() -> dict[str, Any]:
    index = get_index()
    return {
        "status": "ok",
        "attributes": index.available_attributes,
        "dimension": index.dimension,
    }


@app.get("/health")
@app.get("/ping")
@app.get("/")
def root_healthcheck() -> dict[str, Any]:
    """Cloud Run からの疎通確認用にエイリアスを用意。"""
    return healthcheck()


@app.post("/search", response_model=SearchResponse)
def search_endpoint(
    request: SearchRequest,
    index: EmbeddingIndex = Depends(get_index),
) -> SearchResponse:
    if not request.query and request.embedding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query か embedding のいずれかは必須です",
        )

    attributes = request.attributes or index.available_attributes
    attributes = [attr for attr in attributes if attr in index.attribute_to_matrix]
    if not attributes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="指定された属性の埋め込みが見つかりません",
        )

    if request.embedding is not None:
        query_vector = np.asarray(request.embedding, dtype=np.float32)
    else:
        client = get_gemini_client()
        if client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GEMINI_API_KEY が設定されていないため埋め込みを生成できません",
            )
        try:
            query_vector = client.embed(request.query or "")
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini API 呼び出しに失敗しました: {exc}",
            ) from exc

    try:
        hits = index.search(query_vector, attributes, request.top_k)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    results: List[SearchResult] = []
    for hit in hits:
        category = index.get_category(hit["category_id"])
        results.append(
            SearchResult(
                category_id=hit["category_id"],
                attribute=hit["attribute"],
                score=hit["score"],
                text=hit["text"],
                category=category,
            )
        )

    attribute_counts = Counter(hit.attribute for hit in results)
    meta = SearchMetadata(
        top_k=request.top_k,
        attributes=attributes,
        embedding_source=request.embedding_source(),
        result_count=len(results),
        attribute_counts=dict(attribute_counts),
    )

    return SearchResponse(results=results, meta=meta)


@app.get("/attributes", response_model=RootModel[List[str]])
def list_attributes(index: EmbeddingIndex = Depends(get_index)) -> RootModel[List[str]]:
    return RootModel(index.available_attributes)
