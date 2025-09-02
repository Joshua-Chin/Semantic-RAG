from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from rag.util import cleanup
from rag.embed import get_document_contents, get_query_strings


CONTEXTUAL_RERANK_V2_2B = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b"

def rerank(benchmark, documents, ranks, model_path=CONTEXTUAL_RERANK_V2_2B, topk=32):
    reranker = Reranker(model_path)
    reranks = []
    for test, doc_idxs in tqdm(
        zip(benchmark, ranks),
        total=min(len(benchmark), len(ranks))
    ):
        top_documents = [documents[doc_idx] for doc_idx in doc_idxs[:topk]]
        result = reranker(
            query=get_query_strings([test])[0],
            instruction='',
            documents=get_document_contents(top_documents),
        )
        rerank = [int(doc_idxs[idx]) for _, idx, _ in result]
        reranks.append(rerank)
    return reranks

class Reranker:

    def __init__(self, model_path: str=CONTEXTUAL_RERANK_V2_2B):
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # so -1 is the real last token for all prompts
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            dtype=dtype,
        )
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, query: str, instruction: str, documents: list[str]):
        results = infer_w_hf(self.model, self.tokenizer, query, instruction, documents)
        cleanup()
        return results

def format_prompts(query: str, instruction: str, documents: list[str]) -> list[str]:
    """Format query and documents into prompts for reranking."""
    if instruction:
        instruction = f" {instruction}"
    prompts = []
    for doc in documents:
        prompt = f"Check whether a given document contains information helpful to answer the query.\n<Document> {doc}\n<Query> {query}{instruction} ??"
        prompts.append(prompt)
    return prompts

def infer_w_hf(model, tokenizer, query: str, instruction: str, documents: list[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = format_prompts(query, instruction, documents)
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    next_logits = out.logits[:, -1, :]  # [batch, vocab]

    scores_bf16 = next_logits[:, 0].to(torch.bfloat16)
    scores = scores_bf16.float().tolist()

    # Sort by score (descending)
    results = sorted([(s, i, documents[i]) for i, s in enumerate(scores)], key=lambda x: x[0], reverse=True)
    return results