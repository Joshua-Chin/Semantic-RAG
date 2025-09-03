import torch
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig

from rag.load import load_benchmark_corpus, corpus_to_texts_metadatas
from rag.chunk import create_documents
from rag.embed import compute_similarities, get_query_strings, get_document_contents
from rag.rerank import rerank
from rag.metrics import print_evaluations
from rag.util import cleanup


def main():
    # Load LegalBench-RAG
    print("Loading corpus (privacy_qa)")
    benchmark, corpus = load_benchmark_corpus()
    texts, metadatas = corpus_to_texts_metadatas(corpus)
    # Chunk texts
    print("Loading model")
    embedding_model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        model_kwargs={
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        }
    )
    print("Chunking text")
    documents = create_documents(
        embedding_model,
        texts, metadatas,
        chunk_size=500,
        atom_size=200,
    )
    # Embed documents and compute similarities
    print("Embedding chunks")
    similarities = compute_similarities(
        embedding_model,
        queries=get_query_strings(benchmark),
        documents=get_document_contents(documents),
    )
    # Cleanup embedding model
    del embedding_model
    cleanup()
    # Evaluate baseline
    print("Semantic Chunker + Embedding Evaluation")
    ranks = torch.argsort(similarities, descending=True)
    print_evaluations(benchmark, documents, ranks)
    # Rerank TOP_K documents
    TOP_K = 64
    print(f"Reranking top {TOP_K} documents")
    reranks = rerank(
        benchmark, documents, ranks,
        model_path="ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b",
        topk=TOP_K
    )
    # Evaluate reranked
    print(f"Semantic Chunker + Embedding + Reranker (K={TOP_K}) Evaluation")
    print_evaluations(benchmark, documents, reranks)

if __name__ == '__main__':
    main()