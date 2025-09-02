# Semantic RAG: A Deep Dive into Advanced Chunking for Legal Document Analysis

This project prototypes and evaluates a state-of-the-art Retrieval-Augmented Generation (RAG) pipeline designed to tackle the unique challenges of dense, domain-specific legal text.

The core improvement is a **Recursive Semantic Chunker**, an algorithm that recursively partition documents based on their underlying semantic structure, leading to significant improvements in retrieval accuracy.

The pipeline was developed and validated on the challenging **LegalBench-RAG** benchmark, using the `privacy_qa` and `maud` (Mergers & Acquisitions) datasets.

## Key Achievements

*   **State-of-the-Art Performance:** Achieved a robust and reproducible **Precision@1 of 0.3503** on the `privacy_qa` dataset, an **18% relative improvement** over a strong reranker baseline that uses a traditional `RecursiveCharacterTextSplitter`.
*   **Advanced Chunking Algorithm:** Designed and implemented the **Recursive Semantic Chunker**, a recursive algorithm that intelligently partitions documents by repeatedly splitting at the point of maximum semantic distance.
*   **Robust Preprocessing:** Developed a domain-aware preprocessing pipeline using linguistic rules to create a clean, high-quality stream of complete sentences for the chunking algorithm.
*   **Validated Robustness:** Successfully demonstrated the pipeline's strong performance on the complex `maud-mini` dataset, proving the system's ability to generalize across different styles of legal text.

## Performance Results

The final pipeline, featuring the **Recursive Semantic Chunker**, was benchmarked against the original paper's results and a strong, reranked baseline. This "Strong Baseline" is a powerful RAG pipeline in its own right, using the same state-of-the-art embedding and reranking models as the final version. This ensures a direct, apples-to-apples comparison where the only significant variable is the chunking strategy.

The results below are from the final, reproducible configuration, demonstrating a clear and significant improvement at each stage.

**Dataset: `privacy_qa`**

| Metric          | Original Paper (Best) | Strong Baseline (Reranked) | **Final Model (Semantic Chunker)** |
| :------------   | :-------------------- | :------------------------- | :--------------------------------- |
| **Precision@1** | 0.1394                | 0.2968                     | **0.3503 (+18.0%)**                |
| **Recall@1**    | 0.0732                | 0.1806                     | **0.1872 (+3.7%)**                 |
| **PR AUC**      | N/A                   | 0.1380                     | **0.1491 (+8.0%)**                 |
| Precision@8     | 0.0957                | 0.1603                     | **0.1868 (+16.5%)**                |
| Recall@64       | 0.7961                | **0.9538**                 | 0.9118                             |

## Models, Data, and Licensing

This project leverages several state-of-the-art models and datasets. Full attribution is provided below. This project is for personal, non-commercial educational and demonstration purposes.

*   **LegalBench-RAG Dataset:**
    *   **Source:** [LegalBench on Github](https://github.com/ZeroEntropy-AI/legalbenchrag)
    *   **License:** `CC-BY-NC-SA-4.0`. The dataset is used here in accordance with its non-commercial and attribution requirements.
*   **Embedding Models (`Qwen3-Embedding-8B`):**
    *   **Source:** [Qwen on Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
    *   **License:** `Apache-2.0`, which permits commercial use.
*   **Reranker Model (`ctxl-rerank-v2-instruct`):**
    *   **Source:** [ContextualAI on Hugging Face](https://huggingface.co/ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b)
    *   **License:** `CC-BY-NC-SA-4.0`. This model is used in accordance with its non-commercial and attribution requirements.

## Setup & Usage

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Joshua-Chin/Semantic-RAG.git
    cd Semantic-RAG
    ```

2.  **Set up the Python environment:**
    *   **Crucial Step:** First, install PyTorch by following the official instructions for your hardware at [pytorch.org](https://pytorch.org/).
    *   Next, install the project and all dependencies from the `pyproject.toml` file:
        ```bash
        pip install -e .
        ```

### Running the Final Evaluation

To reproduce the final, optimized results shown above, run the main evaluation script from the root directory:

```bash
python evaluate.py
```

## Project Structure
```
.
├── evaluate.py                 # Main script to reproduce final results
│
├── notebooks/                  # Each notebook corresponds a RAG pipeline + evaulation
│   ├── 01-Baseline.ipynb       # Baseline RAG using the `Qwen-Embedding-8B` model
│   ├── 02-Filenames.ipynb      # Adds file metadata to the RAG documents
│   ├── 03-Reranker.ipynb       # Adds reranking with `Contextual AI Reranker v2 2B` model
│   ├── 04-MAUD.ipynb           # Runs evaluation of (03) on Mergers & Acquisitions corpus
│   ├── 05a-Chunking.ipynb      # Visualizes the chunks generated by RCTS 
│   ├── 05b-Semantic.ipynb      # Visualizes the chunks generated by the semantic splitter
│   ├── 05c-Eval.ipynb          # Adds semantic chunking to (03)
│   └── 08-Semantic-MAUD.ipynb  # Runs evaluation of (05c) on Mergers & Acquisitions corpus
│
└── rag/
    ├── chunk.py                # Contains the semantic chunking algorithms
    ├── embed.py                # Logic for loading and using embedding models
    ├── load.py                 # Data downloading and loading utilities for LegalBench-RAG
    ├── metrics.py              # Evaluation metric calculations
    ├── rerank.py               # Logic for loading and using the reranker
    └── util.py                 # Helper functions (e.g., clearing GPU cache)
``` 