from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from dinov2_inference import (
    DINOv2Embedder,
    EmbeddingRecord,
    FewShotMatch,
    FewShotMatcher,
    InMemoryEmbeddingStore,
    MatchResult,
    DetectionHint,
    _compute_agreement,
    similarity_tier,
    _SIMILARITY_HIGH,
    _SIMILARITY_MED,
    _SIMILARITY_LOW,
)


# ─── helpers ──────────────────────────────────────────────────────────────────


def make_image(width=512, height=512) -> Image.Image:
    return Image.new("RGB", (width, height), color=(240, 240, 240))


def rand_unit_vector(dim=768) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_record(tenant_id="t1", class_slug="sprinkler-pendent-k57", sample_id=None) -> EmbeddingRecord:
    return EmbeddingRecord(
        tenant_id=tenant_id,
        class_slug=class_slug,
        sample_id=sample_id or str(uuid4()),
        embedding=rand_unit_vector(),
    )


def make_hint(
    yolo_class="sprinkler-pendent-k57",
    bbox=(100, 100, 140, 140),
) -> DetectionHint:
    return DetectionHint(
        detection_id=str(uuid4()),
        tile_id="tile-001",
        yolo_class_slug=yolo_class,
        bbox_x1=bbox[0], bbox_y1=bbox[1],
        bbox_x2=bbox[2], bbox_y2=bbox[3],
        yolo_confidence=0.85,
    )


def make_mock_embedder(embedding: np.ndarray) -> MagicMock:
    embedder = MagicMock(spec=DINOv2Embedder)
    embedder.embed.return_value = embedding
    embedder.embed_batch.return_value = [embedding]
    return embedder


# ─── InMemoryEmbeddingStore ───────────────────────────────────────────────────


class TestInMemoryEmbeddingStore:
    def test_upsert_and_count(self):
        store = InMemoryEmbeddingStore()
        store.upsert(make_record(tenant_id="t1"))
        store.upsert(make_record(tenant_id="t1"))
        assert store.count_by_tenant("t1") == 2

    def test_upsert_replaces_same_sample_id(self):
        store = InMemoryEmbeddingStore()
        record = make_record(sample_id="fixed-id")
        store.upsert(record)

        updated = EmbeddingRecord(
            tenant_id=record.tenant_id,
            class_slug="extintor-pqs",
            sample_id="fixed-id",
            embedding=rand_unit_vector(),
        )
        store.upsert(updated)

        assert store.count_by_tenant(record.tenant_id) == 1
        results = store.query_topk(record.tenant_id, updated.embedding, k=1)
        assert results[0].class_slug == "extintor-pqs"

    def test_tenant_isolation(self):
        store = InMemoryEmbeddingStore()
        store.upsert(make_record(tenant_id="t1"))
        store.upsert(make_record(tenant_id="t2"))
        assert store.count_by_tenant("t1") == 1
        assert store.count_by_tenant("t2") == 1
        assert store.count_by_tenant("t3") == 0

    def test_query_returns_sorted_by_similarity(self):
        store = InMemoryEmbeddingStore()
        base = rand_unit_vector()

        # Insert 3 records with different distances from base
        close = EmbeddingRecord("t1", "class-close", "s1", base.copy())
        far = EmbeddingRecord("t1", "class-far", "s2", rand_unit_vector())
        store.upsert(close)
        store.upsert(far)

        results = store.query_topk("t1", base, k=2)
        assert results[0].similarity >= results[1].similarity
        assert results[0].class_slug == "class-close"

    def test_query_respects_k_limit(self):
        store = InMemoryEmbeddingStore()
        for i in range(10):
            store.upsert(make_record(tenant_id="t1", sample_id=str(i)))
        results = store.query_topk("t1", rand_unit_vector(), k=3)
        assert len(results) == 3

    def test_query_empty_store_returns_empty(self):
        store = InMemoryEmbeddingStore()
        results = store.query_topk("t1", rand_unit_vector(), k=5)
        assert results == []

    def test_delete_by_class(self):
        store = InMemoryEmbeddingStore()
        store.upsert(make_record(tenant_id="t1", class_slug="class-a", sample_id="s1"))
        store.upsert(make_record(tenant_id="t1", class_slug="class-a", sample_id="s2"))
        store.upsert(make_record(tenant_id="t1", class_slug="class-b", sample_id="s3"))

        deleted = store.delete_by_class("t1", "class-a")
        assert deleted == 2
        assert store.count_by_tenant("t1") == 1

    def test_delete_by_class_cross_tenant_safe(self):
        store = InMemoryEmbeddingStore()
        store.upsert(make_record(tenant_id="t1", class_slug="class-a", sample_id="s1"))
        store.upsert(make_record(tenant_id="t2", class_slug="class-a", sample_id="s2"))

        store.delete_by_class("t1", "class-a")
        assert store.count_by_tenant("t2") == 1  # t2 not affected

    def test_similarity_is_cosine(self):
        store = InMemoryEmbeddingStore()
        v = rand_unit_vector()
        store.upsert(EmbeddingRecord("t1", "cls", "s1", v))
        results = store.query_topk("t1", v, k=1)
        assert abs(results[0].similarity - 1.0) < 1e-5  # same vector → cosine = 1.0


# ─── DINOv2Embedder.make_crop ─────────────────────────────────────────────────


class TestMakeCrop:
    def test_centered_crop_correct_size(self):
        img = make_image(512, 512)
        crop = DINOv2Embedder.make_crop(img, 200, 200, 240, 240, dinov2_window=128)
        assert crop.size == (128, 128)

    def test_pads_when_near_edge(self):
        img = make_image(512, 512)
        # bbox near top-left corner
        crop = DINOv2Embedder.make_crop(img, 5, 5, 30, 30, dinov2_window=128)
        assert crop.size == (128, 128)

    def test_pads_when_near_bottom_right(self):
        img = make_image(200, 200)
        crop = DINOv2Embedder.make_crop(img, 180, 180, 200, 200, dinov2_window=128)
        assert crop.size == (128, 128)

    def test_different_window_sizes(self):
        img = make_image(512, 512)
        for window in [96, 112, 128, 160]:
            crop = DINOv2Embedder.make_crop(img, 250, 250, 280, 280, dinov2_window=window)
            assert crop.size == (window, window)


# ─── FewShotMatcher ───────────────────────────────────────────────────────────


class TestFewShotMatcher:
    def _make_matcher(self, embedding=None, store=None):
        emb = embedding if embedding is not None else rand_unit_vector()
        embedder = make_mock_embedder(emb)
        embedder.make_crop = DINOv2Embedder.make_crop
        s = store or InMemoryEmbeddingStore()
        return FewShotMatcher(embedder=embedder, store=s), s, emb

    def test_returns_none_for_empty_library(self):
        matcher, store, emb = self._make_matcher()
        result = matcher.match("t1", make_image(), make_hint())
        assert result is None

    def test_returns_match_for_similar_embedding(self):
        query_emb = rand_unit_vector()
        matcher, store, _ = self._make_matcher(embedding=query_emb)

        # Store a record with the same embedding (similarity = 1.0)
        store.upsert(EmbeddingRecord("t1", "sprinkler-pendent-k57", "s1", query_emb.copy()))

        result = matcher.match("t1", make_image(), make_hint())
        assert result is not None
        assert result.matched_class_slug == "sprinkler-pendent-k57"
        assert abs(result.similarity - 1.0) < 1e-4

    def test_returns_none_below_minimum_threshold(self):
        query_emb = rand_unit_vector()
        matcher, store, _ = self._make_matcher(embedding=query_emb)

        # Store orthogonal vector (cosine similarity ≈ 0)
        orthogonal = rand_unit_vector()
        orthogonal -= orthogonal.dot(query_emb) * query_emb
        orthogonal /= np.linalg.norm(orthogonal)
        store.upsert(EmbeddingRecord("t1", "some-class", "s1", orthogonal))

        result = matcher.match("t1", make_image(), make_hint())
        assert result is None

    def test_agreement_when_yolo_and_dinov2_match_same_class(self):
        query_emb = rand_unit_vector()
        matcher, store, _ = self._make_matcher(embedding=query_emb)
        store.upsert(EmbeddingRecord("t1", "sprinkler-pendent-k57", "s1", query_emb.copy()))

        result = matcher.match("t1", make_image(), make_hint(yolo_class="sprinkler-pendent-k57"))
        assert result.agreement == "agree"

    def test_disagreement_when_yolo_and_dinov2_differ(self):
        query_emb = rand_unit_vector()
        matcher, store, _ = self._make_matcher(embedding=query_emb)
        store.upsert(EmbeddingRecord("t1", "extintor-pqs", "s1", query_emb.copy()))

        result = matcher.match("t1", make_image(), make_hint(yolo_class="sprinkler-pendent-k57"))
        assert result.agreement == "disagree"
        assert result.matched_class_slug == "extintor-pqs"

    def test_batch_returns_same_count_as_input(self):
        query_emb = rand_unit_vector()
        matcher, store, _ = self._make_matcher(embedding=query_emb)
        store.upsert(EmbeddingRecord("t1", "cls", "s1", query_emb.copy()))

        # Mock embed_batch to return embeddings for all hints
        matcher._embedder.embed_batch.return_value = [query_emb] * 3

        hints = [make_hint() for _ in range(3)]
        results = matcher.match_batch("t1", make_image(), hints)
        assert len(results) == 3

    def test_batch_returns_nones_for_empty_library(self):
        matcher, _, _ = self._make_matcher()
        hints = [make_hint() for _ in range(3)]
        results = matcher.match_batch("t1", make_image(), hints)
        assert all(r is None for r in results)

    def test_embed_and_store_persists_record(self):
        matcher, store, _ = self._make_matcher()
        crop = make_image(128, 128)
        record = matcher.embed_and_store("t1", "sprinkler-pendent-k57", crop, "sample-001")

        assert record.sample_id == "sample-001"
        assert record.class_slug == "sprinkler-pendent-k57"
        assert store.count_by_tenant("t1") == 1

    def test_embed_and_store_generates_uuid_if_no_sample_id(self):
        matcher, store, _ = self._make_matcher()
        record = matcher.embed_and_store("t1", "extintor-pqs", make_image(128, 128))
        from uuid import UUID
        UUID(record.sample_id)  # raises if not valid UUID


# ─── helpers ──────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_compute_agreement_same_class(self):
        assert _compute_agreement("cls-a", "cls-a") == "agree"

    def test_compute_agreement_different_class(self):
        assert _compute_agreement("cls-a", "cls-b") == "disagree"

    @pytest.mark.parametrize("sim,tier", [
        (0.95, "high"), (0.90, "high"),
        (0.80, "medium"), (0.75, "medium"),
        (0.65, "low"), (0.60, "low"),
        (0.50, "no_match"), (0.10, "no_match"),
    ])
    def test_similarity_tier(self, sim, tier):
        assert similarity_tier(sim) == tier
