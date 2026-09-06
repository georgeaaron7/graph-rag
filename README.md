# GraphRAG Knowledge Assistant

## demo
https://github.com/user-attachments/assets/5660e50c-4ad8-4023-991b-219e37debdc0

## about 
a question-answering system over your PDFs that combines a **neo4j knowledge
graph** with **faiss vector search**, fuses the two rankings with **reciprocal
rank fusion**, and answers with **inline citations** back to the source page.


## but why build a graph RAG pipeline at all? 
vector search retrieves by similarity; it can't traverse relationships. Multi-hop questions ("which company did the founder of X later join?") need the graph.

## techstack
Python · FastAPI · Neo4j · FAISS · sentence-transformers · OpenAI-compatible LLM
client (OpenAI / local vLLM / Ollama / vast.ai) · Next.js + TypeScript frontend.

## dataset for eval 
[`yixuantt/MultiHopRAG`](https://huggingface.co/datasets/yixuantt/MultiHopRAG) (ODC-BY,
COLM 2024, [arXiv:2401.15391](https://arxiv.org/abs/2401.15391)) ~2,556 queries whose
supporting evidence is spread across **multiple** news articles, shipped with the article
corpus.

## results 
improved Hit@5 by 23.4% and MRR by 18.2% over vector-only RAG on a 500-question eval

