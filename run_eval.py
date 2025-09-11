# save as run_eval.py
import csv, requests, time

IN = "eval_questions.csv"
OUT = "eval_answers.csv"

with open(IN, newline='', encoding="utf-8") as f, open(OUT, "w", newline='', encoding="utf-8") as g:
    r = csv.DictReader(f)
    w = csv.DictWriter(g, fieldnames=["id","question","answer","citations"])
    w.writeheader()
    for row in r:
        qid, q = row["id"], row["question"]
        t0 = time.time()
        resp = requests.get("http://localhost:8000/ask", params={"q": q}, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        w.writerow({
            "id": qid,
            "question": q,
            "answer": data.get("answer",""),
            "citations": " | ".join(c.get("path","") for c in data.get("citations", []))
        })
        print(f"[{qid}] {q}  ({time.time()-t0:.1f}s)")
print("Wrote", OUT)
