"""unit tests for the recursive chunker (dependency-free, offline-runnable)."""
import unittest
from app.ingestion.chunking import chunk_document, split_text


class TestChunking(unittest.TestCase):
    def test_respects_chunk_size(self):
        text = "word " * 1000
        chunks = split_text(text, chunk_size=200, chunk_overlap=20)
        # each chunk is at most chunk_size + overlap by construction
        self.assertTrue(all(len(c) <= 200 + 20 for c in chunks))
        self.assertGreater(len(chunks), 1)

    def test_returns_something_for_short_text(self):
        chunks = split_text("A short sentence.", chunk_size=800, chunk_overlap=120)
        self.assertEqual(chunks, ["A short sentence."])

    def test_chunk_ids_deterministic_and_prefixed(self):
        kwargs = dict(source="doc.pdf", page=3, chunk_size=80, chunk_overlap=10)
        a = chunk_document("hello world. " * 50, **kwargs)
        b = chunk_document("hello world. " * 50, **kwargs)
        self.assertEqual(
            [c.metadata["chunk_id"] for c in a],
            [c.metadata["chunk_id"] for c in b],
        )
        self.assertTrue(a[0].metadata["chunk_id"].startswith("doc.pdf::p3::c0"))

    def test_overlap_must_be_smaller_than_size(self):
        with self.assertRaises(ValueError):
            split_text("x", chunk_size=10, chunk_overlap=10)


if __name__ == "__main__":
    unittest.main()
