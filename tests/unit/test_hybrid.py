"""Tests for the pure fusion functions in src/retrieval/hybrid.py.

reciprocal_rank_fusion and weighted_score_fusion are stateless —
no embedder, no Qdrant, no network. These are the highest-value tests
in the suite: the logic is subtle (RRF score formula, normalisation,
tie-breaking) and bugs here would be completely silent at the pipeline level.
"""
from __future__ import annotations

import pytest

from src.retrieval.hybrid import reciprocal_rank_fusion, weighted_score_fusion
from tests.conftest import make_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(filename: str, chunk_index: int = 0, score: float = 0.9, text: str = "") -> object:
    return make_result(
        chunk_text=text or f"text from {filename}:{chunk_index}",
        score=score,
        filename=filename,
        chunk_index=chunk_index,
    )


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------

class TestRRF:
    def test_top_ranked_in_both_lists_wins(self):
        winner = _r("a.pdf", 0, score=0.9)
        other  = _r("b.pdf", 0, score=0.8)
        result = reciprocal_rank_fusion([winner, other], [winner, other], k=60)
        assert result[0].chunk_text == winner.chunk_text

    def test_rrf_score_formula(self):
        # Single item ranked #1 in both lists with k=60
        # Expected score: 1/(60+1) + 1/(60+1) = 2/61
        r = _r("a.pdf", 0)
        result = reciprocal_rank_fusion([r], [r], k=60)
        assert len(result) == 1
        assert abs(result[0].score - 2 / 61) < 1e-9

    def test_k_parameter_affects_score(self):
        r = _r("a.pdf", 0)
        res_k1  = reciprocal_rank_fusion([r], [r], k=1)
        res_k60 = reciprocal_rank_fusion([r], [r], k=60)
        # Lower k → higher score for rank-1 item
        assert res_k1[0].score > res_k60[0].score

    def test_no_overlap_all_items_returned(self):
        r1 = _r("a.pdf", 0)
        r2 = _r("b.pdf", 0)
        r3 = _r("c.pdf", 0)
        result = reciprocal_rank_fusion([r1, r2], [r3], k=60)
        assert len(result) == 3

    def test_no_overlap_rank1_sparse_beats_rank2_dense(self):
        # r3 is rank-1 in sparse (score 1/61); r2 is rank-2 in dense (score 1/62)
        r1 = _r("a.pdf", 0)
        r2 = _r("b.pdf", 0)
        r3 = _r("c.pdf", 0)
        result = reciprocal_rank_fusion([r1, r2], [r3], k=60)
        texts = [r.chunk_text for r in result]
        # r1 gets 1/61 from dense, r3 gets 1/61 from sparse — equal score, order may vary
        # r2 gets 1/62 from dense — lower than r3
        r2_pos = texts.index(r2.chunk_text)
        r3_pos = texts.index(r3.chunk_text)
        assert r3_pos < r2_pos

    def test_retrieval_method_tagged_hybrid_rrf(self):
        r = _r("a.pdf", 0)
        result = reciprocal_rank_fusion([r], [r])
        assert all(x.retrieval_method == "hybrid_rrf" for x in result)

    def test_metadata_preserved(self):
        r = _r("myfile.pdf", chunk_index=7)
        result = reciprocal_rank_fusion([r], [r])
        assert result[0].metadata["filename"] == "myfile.pdf"
        assert result[0].metadata["chunk_index"] == 7

    def test_empty_dense_returns_sparse_results(self):
        r = _r("a.pdf", 0)
        result = reciprocal_rank_fusion([], [r], k=60)
        assert len(result) == 1
        assert result[0].chunk_text == r.chunk_text

    def test_empty_sparse_returns_dense_results(self):
        r = _r("a.pdf", 0)
        result = reciprocal_rank_fusion([r], [], k=60)
        assert len(result) == 1

    def test_both_empty_returns_empty(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_scores_sorted_descending(self):
        items = [_r(f"{i}.pdf", 0, score=float(i)) for i in range(5)]
        result = reciprocal_rank_fusion(items, list(reversed(items)), k=60)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_chunk_identity_by_filename_and_index(self):
        # Two results with same filename+index but different text are treated as one chunk
        r1 = make_result(chunk_text="version A", filename="same.pdf", chunk_index=0)
        r2 = make_result(chunk_text="version B", filename="same.pdf", chunk_index=0)
        result = reciprocal_rank_fusion([r1], [r2])
        assert len(result) == 1  # same (filename, chunk_index) key → deduplicated


# ---------------------------------------------------------------------------
# weighted_score_fusion
# ---------------------------------------------------------------------------

class TestWeightedFusion:
    def test_alpha_1_pure_dense(self):
        # alpha=1.0 → sparse contributes nothing; dense order dominates
        dense  = [_r("a.pdf", 0, score=0.9), _r("b.pdf", 0, score=0.5)]
        sparse = [_r("c.pdf", 0, score=100.0)]  # high score but irrelevant with alpha=1
        result = weighted_score_fusion(dense, sparse, alpha=1.0)
        # a.pdf should be #1 (normalised dense score 1.0, sparse 0)
        # c.pdf gets dense score 0 (absent) and sparse score 1.0, but scaled by 0
        assert result[0].metadata["filename"] == "a.pdf"

    def test_alpha_0_pure_sparse(self):
        dense  = [_r("a.pdf", 0, score=100.0)]  # high dense, irrelevant with alpha=0
        sparse = [_r("b.pdf", 0, score=0.9), _r("c.pdf", 0, score=0.1)]
        result = weighted_score_fusion(dense, sparse, alpha=0.0)
        assert result[0].metadata["filename"] == "b.pdf"

    def test_retrieval_method_tagged_hybrid_weighted(self):
        r = _r("a.pdf", 0)
        result = weighted_score_fusion([r], [r])
        assert all(x.retrieval_method == "hybrid_weighted" for x in result)

    def test_all_items_included(self):
        dense  = [_r("a.pdf", 0), _r("b.pdf", 0)]
        sparse = [_r("c.pdf", 0)]
        result = weighted_score_fusion(dense, sparse)
        assert len(result) == 3

    def test_overlapping_items_not_duplicated(self):
        r = _r("a.pdf", 0)
        result = weighted_score_fusion([r], [r])
        assert len(result) == 1

    def test_scores_normalised_between_0_and_1(self):
        dense  = [_r("a.pdf", 0, score=100.0), _r("b.pdf", 0, score=50.0)]
        sparse = [_r("a.pdf", 0, score=200.0), _r("b.pdf", 0, score=100.0)]
        result = weighted_score_fusion(dense, sparse, alpha=0.5)
        for r in result:
            assert 0.0 <= r.score <= 1.0 + 1e-9

    def test_single_item_in_each_list_normalises_to_zero(self):
        # When a list has only one item, min-max normalisation gives 0:
        # denom = max - min = 0, falls back to 1.0, so (score - score) / 1.0 = 0.
        # Combined score = alpha*0 + (1-alpha)*0 = 0.  This is expected behaviour —
        # a single item can't be ranked relative to anything.
        dense  = [_r("a.pdf", 0, score=0.7)]
        sparse = [_r("a.pdf", 0, score=55.0)]
        result = weighted_score_fusion(dense, sparse, alpha=0.5)
        assert len(result) == 1
        assert result[0].score == 0.0

    def test_scores_sorted_descending(self):
        dense  = [_r(f"{i}.pdf", 0, score=float(i)) for i in range(4)]
        sparse = [_r(f"{i}.pdf", 0, score=float(3 - i)) for i in range(4)]
        result = weighted_score_fusion(dense, sparse, alpha=0.5)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_dense_uses_only_sparse(self):
        sparse = [_r("a.pdf", 0, score=0.9)]
        result = weighted_score_fusion([], sparse, alpha=0.7)
        assert len(result) == 1
        assert result[0].metadata["filename"] == "a.pdf"

    def test_both_empty_returns_empty(self):
        assert weighted_score_fusion([], []) == []

    def test_metadata_preserved(self):
        r = _r("meta.pdf", chunk_index=5)
        result = weighted_score_fusion([r], [r])
        assert result[0].metadata["filename"] == "meta.pdf"
        assert result[0].metadata["chunk_index"] == 5
