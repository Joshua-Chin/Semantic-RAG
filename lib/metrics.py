import torch


def evaluate_rag(benchmark, documents, similarities, topk):
    precision = recall = 0
    count = 0
    indices = torch.argsort(similarities, descending=True)[:, :topk]
    for test_idx, document_idxs in enumerate(indices):
        # Compute spans
        spans_true = []
        for snippet in benchmark[test_idx]["snippets"]:
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
