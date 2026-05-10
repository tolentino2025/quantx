import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable
from uuid import uuid4

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

_DINOV2_MODEL_NAME = "facebook/dinov2-base"
_EMBEDDING_DIM = 768
_DINOV2_INPUT_SIZE = 224  # model expects 224x224


# ─── data classes ─────────────────────────────────────────────────────────────


@dataclass
class MatchResult:
    sample_id: str
    class_slug: str
    similarity: float  # cosine similarity 0.0–1.0
    tenant_id: str


@dataclass
class FewShotMatch:
    detection_id: str
    tile_id: str
    class_slug: str           # original YOLO class (may differ from match)
    matched_class_slug: str   # top-1 DINOv2 match
    similarity: float
    top_k: list[MatchResult]
    source: str = "dinov2"
    agreement: str = "unknown"  # set by caller after comparing with YOLO


@dataclass
class EmbeddingRecord:
    tenant_id: str
    class_slug: str
    sample_id: str
    embedding: np.ndarray  # shape (768,), L2 normalized


# ─── embedding store protocol ─────────────────────────────────────────────────


@runtime_checkable
class EmbeddingStore(Protocol):
    def upsert(self, record: EmbeddingRecord) -> None: ...
    def query_topk(self, tenant_id: str, embedding: np.ndarray, k: int) -> list[MatchResult]: ...
    def delete_by_class(self, tenant_id: str, class_slug: str) -> int: ...
    def count_by_tenant(self, tenant_id: str) -> int: ...


# ─── in-memory store (testing + dev) ─────────────────────────────────────────


class InMemoryEmbeddingStore:
    """Fast in-memory store for development and unit tests."""

    def __init__(self) -> None:
        self._records: list[EmbeddingRecord] = []

    def upsert(self, record: EmbeddingRecord) -> None:
        # Remove existing record with same sample_id
        self._records = [
            r for r in self._records
            if not (r.tenant_id == record.tenant_id and r.sample_id == record.sample_id)
        ]
        self._records.append(record)

    def query_topk(self, tenant_id: str, embedding: np.ndarray, k: int) -> list[MatchResult]:
        tenant_records = [r for r in self._records if r.tenant_id == tenant_id]
        if not tenant_records:
            return []

        similarities = [
            (r, float(np.dot(embedding, r.embedding)))
            for r in tenant_records
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [
            MatchResult(
                sample_id=r.sample_id,
                class_slug=r.class_slug,
                similarity=sim,
                tenant_id=tenant_id,
            )
            for r, sim in similarities[:k]
        ]

    def delete_by_class(self, tenant_id: str, class_slug: str) -> int:
        before = len(self._records)
        self._records = [
            r for r in self._records
            if not (r.tenant_id == tenant_id and r.class_slug == class_slug)
        ]
        return before - len(self._records)

    def count_by_tenant(self, tenant_id: str) -> int:
        return sum(1 for r in self._records if r.tenant_id == tenant_id)


# ─── pgvector store ───────────────────────────────────────────────────────────


class PostgresEmbeddingStore:
    """Production store backed by pgvector."""

    def __init__(self, dsn: str) -> None:
        import psycopg2  # type: ignore
        from pgvector.psycopg2 import register_vector  # type: ignore
        self._conn = psycopg2.connect(dsn)
        register_vector(self._conn)
        self._ensure_table()

    def _ensure_table(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS symbol_embeddings (
                    tenant_id   TEXT NOT NULL,
                    class_slug  TEXT NOT NULL,
                    sample_id   TEXT NOT NULL,
                    embedding   vector(768) NOT NULL,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (tenant_id, sample_id)
                );
                CREATE INDEX IF NOT EXISTS symbol_embeddings_hnsw
                ON symbol_embeddings
                USING hnsw (embedding vector_cosine_ops);
            """)
            self._conn.commit()

    def upsert(self, record: EmbeddingRecord) -> None:
        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO symbol_embeddings (tenant_id, class_slug, sample_id, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (tenant_id, sample_id)
                DO UPDATE SET
                    class_slug = EXCLUDED.class_slug,
                    embedding  = EXCLUDED.embedding
            """, (record.tenant_id, record.class_slug, record.sample_id, record.embedding))
            self._conn.commit()

    def query_topk(self, tenant_id: str, embedding: np.ndarray, k: int) -> list[MatchResult]:
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT sample_id, class_slug, 1 - (embedding <=> %s::vector) AS similarity
                FROM symbol_embeddings
                WHERE tenant_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding, tenant_id, embedding, k))
            rows = cur.fetchall()

        return [
            MatchResult(
                sample_id=row[0],
                class_slug=row[1],
                similarity=float(row[2]),
                tenant_id=tenant_id,
            )
            for row in rows
        ]

    def delete_by_class(self, tenant_id: str, class_slug: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute("""
                DELETE FROM symbol_embeddings
                WHERE tenant_id = %s AND class_slug = %s
            """, (tenant_id, class_slug))
            deleted = cur.rowcount
            self._conn.commit()
        return deleted

    def count_by_tenant(self, tenant_id: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM symbol_embeddings WHERE tenant_id = %s",
                (tenant_id,),
            )
            return int(cur.fetchone()[0])


# ─── embedder ─────────────────────────────────────────────────────────────────


class DINOv2Embedder:
    """
    Generates L2-normalized CLS token embeddings using DINOv2.
    Cached as a singleton — model is loaded once at startup.
    """

    def __init__(self, model_name: str = _DINOV2_MODEL_NAME) -> None:
        import torch  # type: ignore
        from transformers import AutoImageProcessor, AutoModel  # type: ignore

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading DINOv2 model %s on %s", model_name, self._device)

        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

        log.info("DINOv2 ready — embedding dim: %d", _EMBEDDING_DIM)

    def embed(self, crop: Image.Image) -> np.ndarray:
        return self.embed_batch([crop])[0]

    def embed_batch(self, crops: list[Image.Image]) -> list[np.ndarray]:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        resized = [c.resize((_DINOV2_INPUT_SIZE, _DINOV2_INPUT_SIZE), Image.LANCZOS) for c in crops]
        inputs = self._processor(images=resized, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        cls_tokens = outputs.last_hidden_state[:, 0, :]  # CLS token
        normalized = F.normalize(cls_tokens, dim=-1)
        return [v.cpu().numpy() for v in normalized]

    @staticmethod
    def make_crop(
        image: Image.Image,
        bbox_x1: int, bbox_y1: int, bbox_x2: int, bbox_y2: int,
        dinov2_window: int,
    ) -> Image.Image:
        """
        Extract a square crop centered on the bbox, sized to dinov2_window.
        Pads with white if the crop extends beyond image bounds.
        """
        cx = (bbox_x1 + bbox_x2) // 2
        cy = (bbox_y1 + bbox_y2) // 2
        half = dinov2_window // 2

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(image.width, cx + half)
        y2 = min(image.height, cy + half)

        crop = image.crop((x1, y1, x2, y2))

        if crop.size != (dinov2_window, dinov2_window):
            padded = Image.new("RGB", (dinov2_window, dinov2_window), (255, 255, 255))
            padded.paste(crop, ((dinov2_window - crop.width) // 2, (dinov2_window - crop.height) // 2))
            return padded

        return crop


# ─── few-shot matcher ─────────────────────────────────────────────────────────


# Similarity thresholds (overridable via config/confidence-thresholds.json per class)
_SIMILARITY_HIGH = 0.90
_SIMILARITY_MED = 0.75
_SIMILARITY_LOW = 0.60


@dataclass
class DetectionHint:
    """Minimal detection info needed to match against the personal library."""
    detection_id: str
    tile_id: str
    yolo_class_slug: str
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    yolo_confidence: float


class FewShotMatcher:
    def __init__(self, embedder: DINOv2Embedder, store: EmbeddingStore) -> None:
        self._embedder = embedder
        self._store = store

    def match(
        self,
        tenant_id: str,
        page_image: Image.Image,
        hint: DetectionHint,
        dinov2_window: int = 128,
        top_k: int = 5,
    ) -> Optional[FewShotMatch]:
        """
        Embed the crop and query for top-k matches in the tenant's library.
        Returns None if no matches above the minimum threshold.
        """
        if self._store.count_by_tenant(tenant_id) == 0:
            log.debug("Tenant %s has empty library — skipping DINOv2 match", tenant_id)
            return None

        crop = DINOv2Embedder.make_crop(
            page_image,
            hint.bbox_x1, hint.bbox_y1, hint.bbox_x2, hint.bbox_y2,
            dinov2_window,
        )
        embedding = self._embedder.embed(crop)
        matches = self._store.query_topk(tenant_id, embedding, k=top_k)

        if not matches or matches[0].similarity < _SIMILARITY_LOW:
            return None

        top = matches[0]
        agreement = _compute_agreement(hint.yolo_class_slug, top.class_slug)

        return FewShotMatch(
            detection_id=hint.detection_id,
            tile_id=hint.tile_id,
            class_slug=hint.yolo_class_slug,
            matched_class_slug=top.class_slug,
            similarity=top.similarity,
            top_k=matches,
            agreement=agreement,
        )

    def match_batch(
        self,
        tenant_id: str,
        page_image: Image.Image,
        hints: list[DetectionHint],
        dinov2_window: int = 128,
        top_k: int = 5,
    ) -> list[Optional[FewShotMatch]]:
        if not hints:
            return []

        if self._store.count_by_tenant(tenant_id) == 0:
            log.debug("Tenant %s has empty library — skipping batch", tenant_id)
            return [None] * len(hints)

        crops = [
            DINOv2Embedder.make_crop(
                page_image,
                h.bbox_x1, h.bbox_y1, h.bbox_x2, h.bbox_y2,
                dinov2_window,
            )
            for h in hints
        ]
        embeddings = self._embedder.embed_batch(crops)

        results: list[Optional[FewShotMatch]] = []
        for hint, embedding in zip(hints, embeddings):
            matches = self._store.query_topk(tenant_id, embedding, k=top_k)

            if not matches or matches[0].similarity < _SIMILARITY_LOW:
                results.append(None)
                continue

            top = matches[0]
            results.append(FewShotMatch(
                detection_id=hint.detection_id,
                tile_id=hint.tile_id,
                class_slug=hint.yolo_class_slug,
                matched_class_slug=top.class_slug,
                similarity=top.similarity,
                top_k=matches,
                agreement=_compute_agreement(hint.yolo_class_slug, top.class_slug),
            ))

        return results

    def embed_and_store(
        self,
        tenant_id: str,
        class_slug: str,
        crop: Image.Image,
        sample_id: Optional[str] = None,
    ) -> EmbeddingRecord:
        """Generate and persist an embedding for a new catalog sample."""
        embedding = self._embedder.embed(crop)
        record = EmbeddingRecord(
            tenant_id=tenant_id,
            class_slug=class_slug,
            sample_id=sample_id or str(uuid4()),
            embedding=embedding,
        )
        self._store.upsert(record)
        log.info("Embedding stored | tenant=%s class=%s sample=%s", tenant_id, class_slug, record.sample_id)
        return record


# ─── helpers ──────────────────────────────────────────────────────────────────


def _compute_agreement(yolo_class: str, dinov2_class: str) -> str:
    if yolo_class == dinov2_class:
        return "agree"
    return "disagree"


def similarity_tier(similarity: float) -> str:
    if similarity >= _SIMILARITY_HIGH:
        return "high"
    if similarity >= _SIMILARITY_MED:
        return "medium"
    if similarity >= _SIMILARITY_LOW:
        return "low"
    return "no_match"
