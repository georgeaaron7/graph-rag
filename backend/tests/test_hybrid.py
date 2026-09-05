import unittest

from app.retrieval.hybrid import ScoredChunk, fuse, reciprocal_rank_fusion


class TestHybrid(unittest.TestCase):
    def test_rrf_rewards_higher_ranks(self):
        scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
        self.assertGreater(scores["a"], scores["c"])
        self.assertGreater(scores["b"], scores["d"])

    def test_fuse_tags_retriever_source(self):
        vector = [ScoredChunk("a", "va", 0.9, "vector"), ScoredChunk("b", "vb", 0.8, "vector")]
        graph = [ScoredChunk("a", "ga", 1.0, "graph"), ScoredChunk("c", "gc", 0.5, "graph")]
        out = fuse(vector, graph, top_k=5)
        by_id = {c.chunk_id: c for c in out}
        self.assertEqual(by_id["a"].source, "hybrid")  # found by both
        self.assertEqual(by_id["b"].source, "vector")
        self.assertEqual(by_id["c"].source, "graph")
        self.assertEqual(out[0].chunk_id, "a")  # in both lists -> ranks first

    def test_fuse_respects_top_k(self):
        vector = [ScoredChunk(f"v{i}", "x", 0.5, "vector") for i in range(10)]
        self.assertEqual(len(fuse(vector, [], top_k=3)), 3)

    def test_fuse_keeps_text_from_whichever_hit_has_it(self):
        vector = [ScoredChunk("a", "", 0.9, "vector")]           # no text
        graph = [ScoredChunk("a", "the real text", 1.0, "graph")]
        out = fuse(vector, graph, top_k=1)
        self.assertEqual(out[0].text, "the real text")


if __name__ == "__main__":
    unittest.main()
