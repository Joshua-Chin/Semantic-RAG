import json
import os
from pathlib import Path
import zipfile


ROOT_PATH = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))

DATA_PATH = os.path.join(ROOT_PATH, "data")
ZIP_PATH = os.path.join(DATA_PATH, "LegalBench-RAG.zip")
LEGALBENCH_RAG_PATH = os.path.join(DATA_PATH, "LegalBench-RAG")

def extract_data():
    if not os.listdir(DATA_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            for name in zip_ref.namelist():
                if name.endswith('/'):
                    continue
                zip_ref.extract(name, path=DATA_PATH)

def load_benchmark_corpus(subset="privacy_qa"):
    with open(os.path.join(LEGALBENCH_RAG_PATH, "benchmarks", f"{subset}.json")) as f:
        benchmark = json.load(f)['tests']
    
    corpus = {}
    corpus_path = os.path.join(LEGALBENCH_RAG_PATH, "corpus", subset)
    for document in os.listdir(corpus_path):
        with open(os.path.join(corpus_path, document)) as f:
            corpus[document] = f.read()
    
    return benchmark, corpus