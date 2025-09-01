import json
import os
import random
import zipfile

import requests


ROOT_PATH = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))

DATA_PATH = os.path.join(ROOT_PATH, "data")
ZIP_PATH = os.path.join(DATA_PATH, "LegalBench-RAG.zip")
LEGALBENCH_RAG_PATH = os.path.join(DATA_PATH, "LegalBench-RAG")

DROPBOX_SHARE_LINK = "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=0"

def download_data():
    if not os.path.exists(ZIP_PATH):
        print("Downloading LegalBench-RAG")
        
        direct_download_link = (
            DROPBOX_SHARE_LINK
            .replace("www.dropbox.com", "dl.dropboxusercontent.com")
            .replace("?dl=0", "?dl=0")
        )
        print(f"Downloading from: {direct_download_link}")

        with open(ZIP_PATH, 'wb') as f:
            with requests.get(direct_download_link, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)


def extract_data():
    download_data()
    if not os.listdir(DATA_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            for name in zip_ref.namelist():
                if name.endswith('/'):
                    continue
                zip_ref.extract(name, path=DATA_PATH)

def load_benchmark_corpus(subset="privacy_qa"):
    extract_data()
    with open(os.path.join(LEGALBENCH_RAG_PATH, "benchmarks", f"{subset}.json")) as f:
        benchmark = json.load(f)['tests']
    
    corpus = {}
    corpus_path = os.path.join(LEGALBENCH_RAG_PATH, "corpus", subset)
    for document in os.listdir(corpus_path):
        with open(os.path.join(corpus_path, document)) as f:
            corpus[document] = f.read()
    
    return benchmark, corpus

def load_benchmark_corpus_sample(subset="privacy_qa", num_documents=5):
    extract_data()
    with open(os.path.join(LEGALBENCH_RAG_PATH, "benchmarks", f"{subset}.json")) as f:
        benchmark = json.load(f)['tests']
    
    corpus = {}
    corpus_path = os.path.join(LEGALBENCH_RAG_PATH, "corpus", subset)
    # Sample documents from corpus.
    random.seed(42)
    for document in random.sample(os.listdir(corpus_path), num_documents):
        with open(os.path.join(corpus_path, document)) as f:
            corpus[document] = f.read()
    # restrict tests to the document.
    benchmark_sample = []
    for test in benchmark:
        file_path = test["snippets"][0]["file_path"]
        filename = os.path.basename(file_path)
        if filename in corpus:
            benchmark_sample.append(test)
            
    return benchmark_sample, corpus

def corpus_to_texts_metadatas(corpus):
    names, texts = zip(*corpus.items())
    metadatas = [
        {"source_file": name}
        for idx, name in enumerate(names)
    ]
    return texts, metadatas