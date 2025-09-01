from rag.util import cleanup


def compute_similarities(model, queries, documents):
    document_embeddings = model.encode(
        documents,
        show_progress_bar=True,
    )
    cleanup()
    query_embeddings = model.encode(
        queries,
        prompt_name="query",
        show_progress_bar=True,
    )
    cleanup()
    return model.similarity(query_embeddings, document_embeddings)

def get_document_contents(documents):
    [f"{document.metadata["source_file"]}: {document.page_content}" for document in documents]

def get_query_strings(benchmark):
    return [test['query'] for test in benchmark]