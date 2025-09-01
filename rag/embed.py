from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig

from rag.util import cleanup


def compute_similarities(model_or_path, queries, documents):
    if isinstance(model_or_path, str):
        model = SentenceTransformer(
            model_or_path,
            model_kwargs={
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                "attn_implementation": "flash_attention_2",
            }
        )
    else:
        model = model_or_path

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
    
    similarities = model.similarity(query_embeddings, document_embeddings)
    del model
    cleanup()

    return similarities

def get_document_contents(documents):
    return [f"{document.metadata["source_file"]}: {document.page_content}" for document in documents]

def get_query_strings(benchmark):
    return [test['query'] for test in benchmark]