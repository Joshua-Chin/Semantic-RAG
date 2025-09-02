import re

import numpy as np

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter



def split_to_spans(text, chunk_size):
    """
    Splits a text into spans, first by newline characters unconditionally,
    and then by other sentences for chunks that are too large.
    """
    # Stage 1: Split by newlines
    line_spans = split_to_spans_re('\n', text)

    # Stage 2: For any chunk that is larger than the chunk_size, split by sentence boundaries
    remaining_separators = [
        # Sentence boundaries
        r'(?<=[.?!])\s+',
        # Clause boundaries
        # ';',
        # Other boundaries
        # ':', ',',
    ]
    secondary_splitter = RecursiveCharacterTextSplitter(
        separators=remaining_separators,
        chunk_size=chunk_size,
        chunk_overlap=0,
        add_start_index=True,
        is_separator_regex=True,
    )
    sentence_spans = []
    for start, end in line_spans:
        line_text = text[start:end]
        if len(line_text) <= chunk_size:
            if line_text.strip():
                sentence_spans.append((start, end))
        else:
            doc = Document(page_content=line_text)
            sub_docs = secondary_splitter.split_documents([doc])
            for sub_doc in sub_docs:
                start_relative = sub_doc.metadata['start_index']
                end_relative = start_relative + len(sub_doc.page_content)
                sentence_spans.append((start + start_relative, start + end_relative))
    return sentence_spans

def split_to_spans_re(pattern, text):
    spans = []
    start = 0
    for match in re.finditer(pattern, text):
        if start < match.start():
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans

def window_spans(spans, pad):
    combined_spans = []
    for i in range(len(spans)):
        start = spans[max(0, i - pad)][0]
        end = spans[min(len(spans) - 1, i + pad)][1]
        combined_spans.append((start, end))
    return combined_spans

def pairwise_cosine_distances(embeddings):
    distances = []
    for i in range(len(embeddings) - 1):
        distances.append(1 - embeddings[i] @ embeddings[i + 1])
    return distances

def split_recursive(spans, distances, max_span):
    """Recursively split a span based on cosine distance."""
    distances = np.asarray(distances)
    chunks = []
    def dfs(start, end):
        lo = spans[start][0]
        hi = spans[end - 1][1]
        if (end - start <= 1) or (hi - lo <= max_span):
            chunks.append((lo, hi))
        else:
            split_idx = start + np.argmax(distances[start:end - 1]) + 1
            dfs(start, split_idx)
            dfs(split_idx, end)
    dfs(0, len(spans))
    return chunks