from sklearn.metrics import auc
import torch


def print_evaluations(
    benchmark, documents,
    document_idxs_by_rank,
    topks=(1, 2, 4, 8, 16, 32, 64),
):
    precision_recalls = []
    for k in topks:
        precision, recall = evaluate_rag_reranked(
            benchmark, documents,
            document_idxs_by_rank,
            k
        )
        precision_recalls.append((precision, recall))
        print(f"precision @ {k:<2}: {precision:7.4f}, recall @ {k:<2}: {recall:7.4f}")
    precision_recalls.sort()
    print(f'AUC: {auc(*zip(*precision_recalls))}')

def evaluate_rag(benchmark, documents, similarities, topk):
    document_idxs_by_rank = similarities_to_ranks(similarities[:, :topk])
    return evaluate_rag_reranked(benchmark, documents, document_idxs_by_rank, topk)

def similarities_to_ranks(similarities):
    return torch.argsort(similarities, descending=True)

def evaluate_rag_reranked(benchmark, documents, document_idxs_by_rank, topk):
    precision = recall = 0
    count = 0
    for test, document_idxs in zip(benchmark, document_idxs_by_rank):
        document_idxs = document_idxs[:topk]
        # Compute spans
        spans_true = []
        for snippet in test["snippets"]:
            spans_true.append(snippet["span"])
        spans_pred = []
        for idx in document_idxs:
            document = documents[idx]
            start = document.metadata["start_index"]
            length = len(document.page_content)
            spans_pred.append((start, start + length))
        # Compute precision and recall
        p, r = precision_recall(spans_true, spans_pred)
        # Update accumulators
        precision += p
        recall += r
        count += 1
    return precision / count, recall / count

def precision_recall(spans_true, spans_pred):
    indices_true = as_indices(spans_true)
    indices_pred = as_indices(spans_pred)

    tp = len(indices_true.intersection(indices_pred))
    fp = len(indices_pred.difference(indices_true))
    fn = len(indices_true.difference(indices_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall
    

def as_indices(spans):
    indices = set()
    for start, end in spans:
        indices.update(range(start, end))
    return indices
