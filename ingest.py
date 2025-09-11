import os, json, hashlib, re
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from docx import Document as DocxDocument
from pydantic import BaseModel
# from PyPDF2 import PdfReader  # lightweight PDF text extraction
from pypdf import PdfReader
import requests
import time

# --------- CONFIG (edit these) ----------
print("Setting up config parameters")
ROOTS = [
    r"\\vic-fas1.pfc.forestry.ca\projects_d\Vini\NIR2015_6March2015",
    r"\\vic-fas1.pfc.forestry.ca\projects_d\Vini\NIR2020_1Nov2019",
    r"\\vic-fas1.pfc.forestry.ca\projects_d\Vini\NIR2025_11Dec2024"
]
INCLUDE_EXT = {".pdf", ".docx", ".txt", ".md", ".csv", ".py", ".r"}
EXCLUDE_DIR_NAMES = {".git", "__pycache__", "node_modules", "venv", ".venv"}
CHUNK_SIZE_CHARS = 3000
CHUNK_OVERLAP_CHARS = 300
CHROMA_DIR = "storage/chroma"
COLLECTION_NAME = "local_docs"
CHUNKS_JSONL = "storage/chunks.jsonl"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
# ----------------------------------------
print("Defining functions...")
def embed_texts_ollama(texts):
    # Ollama returns {"embedding": [..]} for single prompt, so do batch simply
    embs = []
    for t in texts:
        r = requests.post(OLLAMA_EMBED_URL, json={"model": OLLAMA_EMBED_MODEL, "prompt": t}, timeout=120)
        r.raise_for_status()
        embs.append(r.json()["embedding"])
    return embs

def iter_files(roots: List[str]):
    for root in roots:
        root_path = Path(root)
        for p in root_path.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in INCLUDE_EXT:
                continue
            if any(name in EXCLUDE_DIR_NAMES for name in p.parts):
                continue
            yield p

def read_text(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext in {".txt", ".md", ".py", ".r", ".csv"}:
            # Simple text read (for CSV we index header + first lines)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                if ext == ".csv":
                    # keep first ~2000 chars to avoid huge tables
                    return f.read(2000)
                return f.read()
        elif ext == ".docx":
            doc = DocxDocument(path)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".pdf":
            reader = PdfReader(str(path))
            texts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                texts.append(t)
            return "\n".join(texts)
        else:
            return ""
    except Exception as e:
        print(f"[read_text] Failed {path}: {e}")
        return ""

def normalize_text(txt: str) -> str:
    # collapse whitespace; keep basic punctuation
    txt = txt.replace("\x00", " ")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def sha256(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8", errors="ignore")).hexdigest()

def chunk_text(txt: str, size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS) -> List[str]:
    chunks = []
    i = 0
    n = len(txt)
    while i < n:
        end = min(i + size, n)
        chunk = txt[i:end]
        chunks.append(chunk)
        if end == n:
            break
        i = end - overlap
        if i < 0: i = 0
    return chunks

def fmt_secs(total_seconds: float) -> str:
    if total_seconds is None or total_seconds != total_seconds:  # NaN guard
        return "estimating..."
    s = int(total_seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

class Chunk(BaseModel):
    id: str
    path: str
    mtime: float
    text: str
print("Defining main()...")
def main():
    os.makedirs("storage", exist_ok=True)
    # client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # try:
    #     collection = client.get_collection(COLLECTION_NAME)
    # except:
    #     collection = client.create_collection(name=COLLECTION_NAME)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    seen_docs = set()  # hash of full doc text for exact dedupe
    all_files = list(iter_files(ROOTS))
    total_files = len(all_files)
    start_time = time.time()
    num_files, num_chunks = 0, 0

    with open(CHUNKS_JSONL, "w", encoding="utf-8") as out:
        for file_num, path in enumerate(all_files, start=1):
            text = read_text(path)
            text = normalize_text(text)
            if not text or len(text) < 50:
                continue

            h = sha256(text)
            if h in seen_docs:
                continue
            seen_docs.add(h)

            stat = path.stat()
            chs = chunk_text(text)

            # progress (counts chunks including this file)
            elapsed = time.time() - start_time
            files_done = file_num
            files_left = max(total_files - files_done, 0)
            rate = (files_done / elapsed) if elapsed > 3 else None  # warm up a few seconds
            eta = (files_left / rate) if rate else None
            print(f"[{files_done}/{total_files}] {path} -> {len(chs)} chunks "
                  f"(chunks so far={num_chunks + len(chs)}) | elapsed={fmt_secs(elapsed)} | ETA={fmt_secs(eta)}")

            if not chs:
                continue

            ids = []
            docs = []
            metas = []
            for idx, ch in enumerate(chs):
                cid = sha256(f"{str(path)}::{idx}")
                ids.append(cid)
                docs.append(ch)
                metas.append({
                    "path": str(path),
                    "mtime": stat.st_mtime,
                    "chunk_index": idx
                })
                # also write a line for BM25/server
                record = Chunk(id=cid, path=str(path), mtime=stat.st_mtime, text=ch)
                out.write(record.model_dump_json() + "\n")
                num_chunks += 1

            # embeddings batch (let Chroma embed externally? we embed here for clarity)
            emb = embed_texts_ollama(docs)
            collection.add(
                ids=ids,
                embeddings=emb,
                metadatas=metas,
                documents=docs
            )
            num_files += 1
            if num_files % 20 == 0:
                print(f"[ingest] files={num_files} chunks={num_chunks}")

    print(f"[done] files={num_files}, chunks={num_chunks}, collection={COLLECTION_NAME}")
print("Running main()...")
if __name__ == "__main__":
    main()
