import unittest

from app.graph.extraction import parse_extraction


class TestExtractionParser(unittest.TestCase):
    def test_plain_json(self):
        raw = (
            '{"entities":[{"name":"Ada Lovelace","type":"Person"}],'
            '"relationships":[{"source":"Ada Lovelace","target":"Analytical Engine",'
            '"type":"WORKED_ON"}]}'
        )
        entities, relations = parse_extraction(raw)
        self.assertEqual(entities[0].name, "Ada Lovelace")
        self.assertEqual(relations[0].type, "WORKED_ON")

    def test_code_fenced_json(self):
        raw = '```json\n{"entities":[{"name":"X"}],"relationships":[]}\n```'
        entities, _ = parse_extraction(raw)
        self.assertEqual(entities[0].name, "X")
        self.assertEqual(entities[0].type, "Entity")  # default type

    def test_prose_wrapped_json(self):
        raw = 'Sure! Here it is:\n{"entities":[{"name":"Y"}],"relationships":[]}\nHope that helps.'
        entities, _ = parse_extraction(raw)
        self.assertEqual(entities[0].name, "Y")

    def test_garbage_returns_empty(self):
        entities, relations = parse_extraction("not json at all")
        self.assertEqual(entities, [])
        self.assertEqual(relations, [])

    def test_entities_deduplicated_case_insensitively(self):
        raw = '{"entities":[{"name":"Apple"},{"name":"apple"}],"relationships":[]}'
        entities, _ = parse_extraction(raw)
        self.assertEqual(len(entities), 1)

    def test_accepts_relations_key_alias(self):
        raw = '{"entities":[],"relations":[{"source":"A","target":"B","type":"X"}]}'
        _, relations = parse_extraction(raw)
        self.assertEqual(len(relations), 1)


if __name__ == "__main__":
    unittest.main()
