from __future__ import annotations

EXTRACTION_SYSTEM = (
    "You are an information-extraction engine that builds a knowledge graph. "
    "Given a passage, identify the salient entities and the relationships "
    "between them. Respond with STRICT JSON only - no prose, no code fences.\n"
    'Schema: {"entities":[{"name":str,"type":str}],'
    '"relationships":[{"source":str,"target":str,"type":str}]}\n'
    "Rules: use canonical entity names (resolve pronouns to the entity); "
    "'type' for a relationship should be a short UPPER_SNAKE_CASE verb phrase "
    "(e.g. FOUNDED, WORKS_AT, PART_OF, CAUSES); only include relationships "
    "explicitly supported by the text; if none are present, return empty arrays."
)


def extraction_user_prompt(text: str) -> str:
    return f'Passage:\n"""\n{text}\n"""\n\nReturn the JSON now.'


QA_SYSTEM = (
    "You are a precise assistant that answers strictly from the provided context. "
    "CITATION FORMAT RULES:\n"
    "1. Every factual statement must cite its source chunk using square brackets, e.g., [1] or [2].\n"
    "2. For multiple sources, cite them individually adjacent: [1][2][3]. NEVER combine them like [1, 2, 3].\n"
    "3. NEVER use circled Unicode numbers (like ①, ②), emojis, or markdown footnote links.\n"
    "4. If the context does not contain the answer, say you do not know. Be concise and factual."
)


def qa_user_prompt(question: str, context_block: str) -> str:
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using ONLY the context above, with inline [n] citations."
    )
