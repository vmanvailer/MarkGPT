import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).parent.resolve()
CHROMA_DIR = str((BASE_DIR / "storage" / "chroma").resolve())
CHUNKS_JSONL = str((BASE_DIR / "storage" / "chunks.jsonl").resolve())
COLLECTION_NAME = "local_docs"

# Retrieval knobs
TOPK_BM25 = 8
TOPK_VEC = 8
TOPK_FUSED = 4
MAX_CHARS_PER_CHUNK = 1200

# Ollama (LLM + embeddings)
# OLLAMA_MODEL = "cas/llama-3.2-3b-instruct"  # e.g., "llama3.2:3b-instruct" if you pulled that tag
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
GEN_TIMEOUT = 300  # seconds

# Target focus for your proof
TARGET_YEARS = {"2015", "2020", "2025"}
KEY_TERMS_KNOWN = [
    "BURNCLASS",
    "recalculation",
    "Residential Firewood",
    "Last Pass",
    "biomass",
    "volume",
]

# Confidence thresholds for smart citations
S_HIGH = 1.2
S_MED = 0.7
S_LOW = 0.4

# =========================
# APP INIT
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Chroma path:", CHROMA_DIR)

# Ping Ollama at startup (best-effort)
try:
    _ = requests.post("http://localhost:11434/api/tags", timeout=5)
except Exception as e:
    print("Warning: Ollama not reachable at startup:", e)

# Connect to Chroma (persistent)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Load chunks for BM25
texts: List[str] = []
ids: List[str] = []
id2meta: Dict[str, Dict] = {}

if not Path(CHUNKS_JSONL).exists():
    raise RuntimeError(f"Missing chunks store: {CHUNKS_JSONL}. Run ingest.py first.")

with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        ids.append(rec["id"])  # type: ignore
        texts.append(rec.get("text", ""))
        id2meta[rec["id"]] = {"path": rec.get("path", ""), "mtime": rec.get("mtime", 0)}

if not texts:
    raise RuntimeError("No chunks loaded from chunks.jsonl; ingestion may have failed.")

bm25 = BM25Okapi([t.split() for t in texts])

# =========================
# HELPERS
# =========================
class Answer(BaseModel):
    answer: str
    citations: List[Dict]

class AskBody(BaseModel):
    q: str
    model: Optional[str] = None  # optional override per request
    strict: Optional[bool] = None  # future toggle
    max_citations: Optional[int] = None  # future cap


def _num(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        return float(x.get("distance", x.get("score", 0.0)))
    try:
        return float(x[0])
    except Exception:
        return 0.0


def embed_query_ollama(q: str) -> List[List[float]]:
    r = requests.post(OLLAMA_EMBED_URL, json={"model": OLLAMA_EMBED_MODEL, "prompt": q}, timeout=60)
    r.raise_for_status()
    return [[float(v) for v in r.json()["embedding"]]]


def ask_llm(prompt: str, model_name: Optional[str] = None) -> str:
    model_to_use = model_name or OLLAMA_MODEL
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model_to_use,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 4096,
                "num_predict": 256,
            },
        },
        timeout=GEN_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()


def build_prompt(question: str, contexts: List[Dict]) -> str:
    header = (
        "You are a precise internal assistant. Answer concisely. "
        "Cite file paths like [PATH: ...]. If unsure, say you don’t know.\n\n"
        f"Question: {question}\n\nContext:\n"
    )
    body = ""
    for i, c in enumerate(contexts, start=1):
        snippet = (c.get("text") or "")[:MAX_CHARS_PER_CHUNK]
        body += f"[{i}] PATH: {c.get('path','')}\n{snippet}\n---\n"
    return header + body + "\nAnswer:"


def infer_focus(q: str, strict_default_nir: bool = False):
    years = set(re.findall(r"\b(20\d{2})\b", q)) & TARGET_YEARS
    report = None
    if re.search(r"\bNIR\b", q, re.I):
        report = "nir"
    elif re.search(r"\bEPR\b", q, re.I):
        report = "epr"
    elif strict_default_nir:
        # When strict is enabled and user didn’t say, assume NIR (your current focus)
        report = "nir"
    return {"years": sorted(years), "report": report}



def path_matches_focus(path: str, focus) -> bool:
    p = (path or "").lower()
    if focus["report"] and focus["report"] not in p:
        return False
    if focus["years"] and not any(y in p for y in focus["years"]):
        return False
    return True


def fuse(bm25_hits, vec_hits):
    scores = {}
    for i, (cid, _) in enumerate(bm25_hits):
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (1 + i))
    for i, (cid, _) in enumerate(vec_hits):
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (1 + i))
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


PII_PATTERNS = [
    r"\b(lunch|dinner|breakfast|home address|phone|email)\b",
    r"\b(where does .* live|what did .* eat|birthday|sin|social insurance)\b",
]


def is_pii_question(q: str) -> bool:
    return any(re.search(p, q, re.I) for p in PII_PATTERNS)


def match_terms(text: str):
    hits = [t for t in KEY_TERMS_KNOWN if re.search(rf"\b{re.escape(t)}\b", text or "", re.I)]
    return hits[:3]


# =========================
# ROUTES
# =========================
@app.get("/ask", response_model=Answer)
def ask_get(q: str = Query(..., description="User question")):
    return _answer_flow(q)


@app.post("/ask", response_model=Answer)
def ask_post(body: AskBody):
    return _answer_flow(body.q, model_override=body.model, strict=body.strict, max_citations=body.max_citations)


def _answer_flow(q: str, model_override: Optional[str] = None, strict: Optional[bool] = None, max_citations: Optional[int] = None) -> Answer:
    # Impossible/PII guard (pre)
    if is_pii_question(q):
        return Answer(answer="I don’t know based on the indexed documents.", citations=[])

    focus = infer_focus(q, strict_default_nir=bool(strict))

    # --- BM25 ---
    toks = q.split()
    scores = bm25.get_scores(toks)
    bm25_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOPK_BM25]
    bm25_hits = [(ids[i], float(scores[i])) for i in bm25_order]

    # strict filter if year/report specified
    bm25_hits = [
        (cid, s) for (cid, s) in bm25_hits if path_matches_focus(id2meta[cid]["path"], focus)
    ] or bm25_hits[:TOPK_BM25]

    # --- Vector (robust) ---
    try:
        qvec = embed_query_ollama(q)
        res = collection.query(
            query_embeddings=qvec,
            n_results=TOPK_VEC,
            include=["documents", "metadatas", "distances"],
        )
        vec_ids = res.get("ids", [[]])[0]
        vec_docs = res.get("documents", [[]])[0]
        vec_meta = res.get("metadatas", [[]])[0]
        vec_dists = [_num(d) for d in res.get("distances", [[]])[0]]
    except Exception:
        vec_ids, vec_docs, vec_meta, vec_dists = [], [], [], []

    if vec_ids:
        filtered = []
        for rid, doc, meta, dist in zip(vec_ids, vec_docs, vec_meta, vec_dists):
            if path_matches_focus(meta.get("path", ""), focus):
                filtered.append((rid, dist, doc, meta))
        if filtered:
            vec_ids, vec_dists, vec_docs, vec_meta = map(list, zip(*filtered))
        else:
            vec_ids, vec_docs, vec_meta, vec_dists = [], [], [], []

    vec_hits = [(rid, 1.0 - dist) for rid, dist in zip(vec_ids, vec_dists)] if vec_ids else []

    # --- Fuse + smart citation count ---
    fused = fuse(bm25_hits, vec_hits)
    top_score = fused[0][1] if fused else 0.0

    if max_citations is not None:
        wanted = max(0, min(int(max_citations), TOPK_FUSED))
    elif top_score >= S_HIGH:
        wanted = min(TOPK_FUSED, 2)
    elif top_score >= S_MED:
        wanted = min(TOPK_FUSED, 3)
    elif top_score >= S_LOW:
        wanted = TOPK_FUSED
    else:
        wanted = 0

    fused_ids = [cid for cid, _ in fused[:wanted]]

    # Build contexts (with reasons)
    vec_map_doc = {rid: doc for rid, doc in zip(vec_ids, vec_docs)}
    vec_map_meta = {rid: m for rid, m in zip(vec_ids, vec_meta)}

    contexts = []
    for cid in fused_ids:
        txt = vec_map_doc.get(cid, texts[ids.index(cid)])
        meta = vec_map_meta.get(cid, id2meta.get(cid, {}))
        reason_terms = match_terms(txt)
        reason = f"keyword(s): {', '.join(reason_terms)}" if reason_terms else "semantic/keyword match"
        contexts.append({"id": cid, "text": txt, "path": meta.get("path", ""), "why": reason})

    # If confidence too low or nothing found -> clean no-answer
    if wanted == 0 or not contexts:
        return Answer(answer="I don’t know based on the indexed documents.", citations=[])

    # LLM prompt
    prompt = build_prompt(q, contexts)

    try:
        llm_answer = ask_llm(prompt, model_override)
        if not llm_answer:
            llm_answer = "I don’t know based on the indexed documents."
        citations_payload = [{"id": c["id"], "path": c["path"], "why": c["why"]} for c in contexts]
        return Answer(answer=llm_answer, citations=citations_payload)
    except requests.exceptions.ReadTimeout:
        # On timeout, return paths only if we had some confidence
        short = "Top matching files:\n" + "\n".join(f"- {c['path']} ({c['why']})" for c in contexts)
        citations_payload = [{"id": c["id"], "path": c["path"], "why": c["why"]} for c in contexts]
        return Answer(answer=short, citations=citations_payload)
    except Exception as e:
        short = "Top matching files:\n" + "\n".join(f"- {c['path']} ({c['why']})" for c in contexts)
        citations_payload = [{"id": c["id"], "path": c["path"], "why": c["why"]} for c in contexts]
        return Answer(answer=short + f"\n\n(Note: generation error: {type(e).__name__})", citations=citations_payload)


# -------------------------
# Minimal UI for local testing
# -------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Local RAG</title>
<style>
  :root{
    --bg:#0e0f13; --panel:#15171c; --panel-2:#1c1f26; --text:#e7e7ea; --muted:#9aa1ac; --accent:#10a37f; --border:#2a2e36;
  }
  @media (prefers-color-scheme: light){
    :root{--bg:#f6f7f9; --panel:#ffffff; --panel-2:#f3f4f7; --text:#0b0c10; --muted:#5b6573; --accent:#10a37f; --border:#e5e7ef;}
  }
  *{box-sizing:border-box}
  body{margin:0; background:var(--bg); color:var(--text); font:15px system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
  .shell{display:grid; grid-template-rows:auto 1fr auto; height:100dvh;}
  header{padding:14px 20px; border-bottom:1px solid var(--border); background:var(--panel); display:flex; justify-content:space-between; align-items:center}
  header .brand{display:flex; gap:10px; align-items:center}
  header .brand .logo{width:28px; height:28px; border-radius:8px; background:var(--accent); display:grid; place-items:center; color:white; font-weight:700}
  header .brand h1{font-size:14px; margin:0; letter-spacing:.3px}
  main{padding:18px; overflow:auto}
  .chat{max-width:900px; margin:0 auto; display:grid; gap:12px}
  .msg{display:flex; gap:10px}
  .msg .bubble{padding:12px 14px; border-radius:14px; background:var(--panel); border:1px solid var(--border); box-shadow:0 1px 0 rgba(0,0,0,.06)}
  .me .bubble{background:var(--panel-2)}
  .small{color:var(--muted); font-size:12px}
  .citations{margin-top:8px}
  .citations a{color:var(--accent); text-decoration:none}
  .composer{padding:14px; border-top:1px solid var(--border); background:var(--panel);}
  .bar{max-width:900px; margin:0 auto; display:flex; gap:10px; align-items:center}
  textarea{flex:1; resize:vertical; min-height:46px; max-height:160px; padding:12px 14px; border-radius:14px; border:1px solid var(--border); background:var(--panel-2); color:var(--text)}
  select,button{border-radius:12px; border:1px solid var(--border); background:var(--panel-2); color:var(--text); padding:10px 12px}
  button.primary{background:var(--accent); color:white; border:0}
</style>
</head>
<body>
<div class=\"shell\">
  <header>
    <div class=\"brand\"><div class=\"logo\">∞</div><h1>Local RAG</h1></div>
    <div class=\"small\">demo</div>
  </header>
  <main>
    <div id=\"chat\" class=\"chat\"></div>
  </main>
  <div class=\"composer\">
    <div class=\"bar\">
      <textarea id=\"q\" placeholder=\"Ask NIR/EPR… (Ctrl+Enter to send)\"></textarea>
      <select id="model">
        <option value=\"llama3.2:3b\">llama3.2:3b (fast)</option>
        <option value\="gpt-oss\">gpt-oss (20B, slower)</option>
      </select>
      <button id=\"send\" class=\"primary\" type=\"button\">Send</button>
      <button id=\"cancel\" type=\"button\">Cancel</button>
      <span id=\"status\" class=\"small\" style=\"margin-left:8px\"></span>
    </div>
    <div class=\"bar small\" style=\"margin-top:8px\">Placeholders: <label style=\"margin-left:6px\"><input type=\"checkbox\" id=\"strict\"> Strict year/report</label> <span style=\"margin-left:10px\">Max citations:</span> <select id=\"maxc\"><option value=\"\">auto</option><option>1</option><option>2</option><option>3</option><option>4</option></select></div>
  </div>
</div>
<script>
const API = location.origin + "/ask";
const chat = document.getElementById("chat");
const qEl = document.getElementById("q");
const sendBtn = document.getElementById("send");
const cancelBtn = document.getElementById("cancel");
const statusEl = document.getElementById("status");

// one controller per request
let inFlightCtrl = null;

function toFileLink(p){
  if(!p) return "";
  // Detect UNC without using backslash string escapes
  const isUNC = p.length>1 && p.charCodeAt(0)===92 && p.charCodeAt(1)===92; // '\\\\'
  // Detect drive-letter path like C:\  (check ':' then backslash by char code)
  const isDrive = /^[A-Za-z]:/.test(p) && p.length>2 && p.charCodeAt(2)===92;

  if(isUNC){
    const without = p.slice(2);               // drop leading \\
    return 'file://' + without.split('\\\\').join('/');
  }
  if(isDrive){
    return 'file:///' + p.split('\\\\').join('/');
  }
  return p;
}

function addMsg(role, html){
  const row = document.createElement("div");
  row.className = "msg "+role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = html;
  row.appendChild(bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
}

async function send(){
  const q = qEl.value.trim();
  if(!q) return;
  const model = document.getElementById("model").value || null;
  const strict = document.getElementById("strict").checked || null;
  const maxcSel = document.getElementById("maxc").value;
  const max_citations = maxcSel ? Number(maxcSel) : null;

  addMsg("me", q.replace(/&/g,"&amp;").replace(/</g,"&lt;"));
  qEl.value = "";

  // ---- status & watchdog ----
  let dots = 0;
  statusEl.textContent = "Thinking";
  sendBtn.disabled = true;
  cancelBtn.disabled = false;

  const tick = setInterval(()=>{
    dots = (dots+1)%4;
    statusEl.textContent = "Thinking" + ".".repeat(dots);
  }, 500);
  const started = Date.now();

  // Optional soft watchdog message if it takes long
  const softTimeoutMs = 90000; // 90s
  const softTimer = setTimeout(()=>{
    statusEl.textContent = "Still working… (retrieval/generation)";
  }, softTimeoutMs);

  // ---- AbortController + hard timeout ----
  // if something is already in flight, abort it first (safety)
  if (inFlightCtrl) { try { inFlightCtrl.abort(); } catch(_){} }
  inFlightCtrl = new AbortController();
  const signal = inFlightCtrl.signal;

  const hardTimeoutMs = 300000; // 5 min hard cap; tweak as you like
  const hardTimer = setTimeout(()=>{
    if (inFlightCtrl) inFlightCtrl.abort("Timed out");
  }, hardTimeoutMs);

  try{
    const res = await fetch(API, {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({q, model, strict, max_citations}),
      signal
    });
    if(!res.ok){
      addMsg("bot", `<div class="small">HTTP ${res.status}</div>`);
      return;
    }
    const data = await res.json();
    let cites = "";
    if(data.citations && data.citations.length){
      cites = `<div class="citations"><div class="small">Citations</div><ul>` +
              data.citations.map(c=>`<li><a href="${toFileLink(c.path)}" target="_blank">${c.path}</a>${c.why?` <span class="small">— ${c.why}</span>`:""}</li>`).join("") +
              "</ul></div>";
    }
    addMsg("bot", `<pre>${(data.answer||"").replace(/</g,"&lt;")}</pre>` + cites);
    const secs = ((Date.now()-started)/1000).toFixed(1);
    statusEl.textContent = `Done in ${secs}s`;
  } catch(e) {
    if (e && (e.name === "AbortError" || String(e).includes("aborted"))) {
      statusEl.textContent = "Canceled";
      // No bot bubble on cancel; uncomment next line if you want one:
      // addMsg("bot", `<div class="small">Request canceled</div>`);
    } else {
      addMsg("bot", `<div class="small">${e.message}</div>`);
      statusEl.textContent = "Error";
    }
  } finally {
    clearInterval(tick);
    clearTimeout(softTimer);
    clearTimeout(hardTimer);
    sendBtn.disabled = false;
    cancelBtn.disabled = true;
    inFlightCtrl = null;
  }
}

cancelBtn.disabled = true; // start disabled

cancelBtn.addEventListener("click", () => {
  if (inFlightCtrl) {
    try { inFlightCtrl.abort("User canceled"); } catch(_) {}
  }
});

// Optional: ESC cancels
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && inFlightCtrl) {
    try { inFlightCtrl.abort("User canceled"); } catch(_) {}
  }
});

sendBtn.addEventListener("click", send);
qEl.addEventListener("keydown", (ev)=>{ if(ev.key==="Enter" && (ev.ctrlKey||ev.metaKey)){ send(); ev.preventDefault(); } });
</script>
</body>
</html>
    """
