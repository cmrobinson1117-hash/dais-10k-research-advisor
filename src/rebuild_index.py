from pathlib import Path
import json, re, hashlib
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def pdf_text(path):
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def parse_name(path):
    name = path.stem
    # sec_filings:tesla_2023
    company, year = name.rsplit("_", 1)
    return company, year

def extract_item_1a(text):
    text = re.sub(r"\s+", " ", text)
    start = re.search(r"Item\s+1A\.?\s*Risk\s+Factors|Item\s+1A\.", text, re.I)
    end = re.search(r"Item\s+1B\.|Item\s+1C\.|Item\s+2\.", text[start.end():], re.I) if start else None

    if not start:
        return ""

    if end:
        return text[start.start(): start.end() + end.start()]
    return text[start.start():]

def make_chunks(text, size=1200, overlap=200):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

docs = []
chunks = []

for pdf in sorted(PDF_DIR.glob("*.pdf")):
    company, year = parse_name(pdf)
    raw = pdf_text(pdf)
    item_1a = extract_item_1a(raw)

    doc_id = hashlib.md5(str(pdf).encode()).hexdigest()
    docs.append({
        "doc_id": doc_id,
        "company": company,
        "year": year,
        "source": str(pdf),
    })

    section_text = item_1a if item_1a else raw[:5000]

    for idx, chunk_text in enumerate(make_chunks(section_text)):
        chunk_id = hashlib.md5(f"{pdf}-{idx}".encode()).hexdigest()
        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "company": company,
            "year": year,
            "section": "item_1a_risk_factors",
            "chunk_index": idx,
            "text": chunk_text,
        })

with open(OUT_DIR / "docs.jsonl", "w") as f:
    for d in docs:
        f.write(json.dumps(d) + "\n")

with open(OUT_DIR / "chunks.jsonl", "w") as f:
    for c in chunks:
        f.write(json.dumps(c) + "\n")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = [c["text"] for c in chunks]
emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype(np.float32))
faiss.write_index(index, str(OUT_DIR / "faiss.index"))

id_map = {str(i): chunks[i]["chunk_id"] for i in range(len(chunks))}
with open(OUT_DIR / "id_map.json", "w") as f:
    json.dump(id_map, f)

print(f"Rebuilt {len(docs)} docs and {len(chunks)} chunks.")