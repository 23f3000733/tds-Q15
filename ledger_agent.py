"""
Ledger Agent: POST {"question": "..."} -> {"answer": ...}

Run locally:   LEDGER_BASE="<your questionData URL ending in &path=/>" \
               OPENAI_API_KEY=... uvicorn ledger_agent:app --port 8000
Inspect data:  python ledger_agent.py --inspect      (prints columns/values; use this first)

Env vars
  LEDGER_BASE      the full API URL from the task (the one ending in path=/)
  OPENAI_API_KEY   key for any OpenAI-compatible endpoint
  OPENAI_BASE_URL  default https://api.openai.com/v1  (e.g. https://aipipe.org/openai/v1)
  LLM_MODEL        default gpt-4o-mini
"""
import json, os, re, sys, time
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import httpx
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------- CONFIG
# Adjust these after running --inspect. Auto-detection is attempted if a name is None/missing.
CFG = dict(
    id_col="id",          # order id (duplicates possible)
    updated_col="updated_at",   # latest wins
    status_col="status",        # 'paid' counts as revenue
    amount_col="amount",        # money column in the order's own currency
    currency_col="currency",    # currency code column
    date_col="created_at",      # business date used for "in March 2026" (change if there is e.g. paid_at/order_date)
    rate_is_usd_per_unit=True,  # True: usd = amount * rate[cur]; False: usd = amount / rate[cur]
    use_export=True,            # True: load /export once; False: paginate /orders
)
TZ = "Asia/Kolkata"
BASE = os.environ.get("LEDGER_BASE", "https://exam.sanand.workers.dev/questionData?email=23f3000733%40ds.study.iitm.ac.in&quizSign=KngAg8U77oMh2SDGF7jlHSciWapMDcih1Gc7r6GryO9BZqvJjFrYmn0KCz214FAXnbIrAV2y4Dv3RCif1CufTGsyTEakwJEKlJt7ZfDDXxjC5KtweJ5eGbOiXqVqgWfcGmuN1e6YGWSK%2BoS4h4mhZt%2F6ZvGxtXOWDpXl7f0MnaL0oVBhGAcLE%2Fxs1K5DfRJ9CGuecD%2BUY59II5Ngs1x3DpfxOOS7qfcZxNDlsOnG3VaXwJi48w%2BWFd75%2BaCk%2FI355o9ZNaA4gyCE7zaXIhLQ4PHbcleP5wQvp5eqhvdIAvq5mKFgwBTDB9haMVUqjOuhxGdojdSafO46oiJY%2BL2hKQ%3D%3D&questionId=q-ledger-agent-server&path=/")
LLM_BASE = os.environ.get("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
LLM_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------- fetching
def with_path(path, **extra):
    u = urlparse(BASE)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q["path"] = path
    q.update({k: str(v) for k, v in extra.items()})
    return urlunparse(u._replace(query=urlencode(q)))

def find_records(obj):
    """Return the first list of dicts found in a JSON response."""
    if isinstance(obj, list):
        return obj if (not obj or isinstance(obj[0], dict)) else []
    if isinstance(obj, dict):
        for v in obj.values():
            r = find_records(v)
            if r:
                return r
    return []

def next_link(obj, cur_page):
    if isinstance(obj, dict):
        for k in ("next", "next_url", "next_page", "nextPage"):
            v = obj.get(k) or (obj.get("links") or {}).get(k) if isinstance(obj.get("links", {}), dict) else obj.get(k)
            if isinstance(v, str) and v.startswith("http"):
                return v
            if isinstance(v, int):
                return with_path("/orders", page=v)
        if obj.get("has_more") or obj.get("hasMore"):
            return with_path("/orders", page=cur_page + 1)
    return None

def fetch_all(client):
    if CFG["use_export"]:
        r = client.get(with_path("/export")); r.raise_for_status()
        try:
            return find_records(r.json())
        except Exception:  # maybe NDJSON
            return [json.loads(l) for l in r.text.splitlines() if l.strip()]
    rows, url, page = [], with_path("/orders", page=1), 1
    while url:
        r = client.get(url); r.raise_for_status()
        j = r.json(); recs = find_records(j)
        if not recs:
            break
        rows += recs
        url = next_link(j, page) or (with_path("/orders", page=page + 1) if "next" not in str(j)[:2000] and len(recs) > 0 and page < 500 and False else None)
        page += 1
    return rows

def fetch_rates(client):
    r = client.get(with_path("/rates")); r.raise_for_status()
    j = r.json()
    def find_map(o):
        if isinstance(o, dict):
            nums = {k: float(v) for k, v in o.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)}
            if nums and len(nums) == len([v for v in o.values() if not isinstance(v, str)]):
                return nums
            for v in o.values():
                m = find_map(v)
                if m:
                    return m
        return {}
    return find_map(j)

# ---------------------------------------------------------------- data prep
DF = None
RATES = {}
SCHEMA_TEXT = ""

def build():
    global DF, RATES, SCHEMA_TEXT
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        rows = fetch_all(c)
        RATES = fetch_rates(c)
    df = pd.json_normalize(rows)
    # latest updated_at wins per order id
    up, idc = CFG["updated_col"], CFG["id_col"]
    if up in df and idc in df:
        df["_u"] = pd.to_datetime(df[up], utc=True, errors="coerce")
        df = df.sort_values("_u").drop_duplicates(idc, keep="last").drop(columns="_u")
    # business date in IST
    dc = CFG["date_col"]
    if dc in df:
        ts = pd.to_datetime(df[dc], utc=True, errors="coerce").dt.tz_convert(TZ)
        df["date_ist"] = ts.dt.tz_localize(None)
        df["year"], df["month"], df["day"] = ts.dt.year, ts.dt.month, ts.dt.day
    # USD conversion
    ac, cc = CFG["amount_col"], CFG["currency_col"]
    if ac in df:
        amt = pd.to_numeric(df[ac], errors="coerce")
        if cc in df and RATES:
            rate = df[cc].map(RATES).fillna(1.0)
            df["amount_usd"] = amt * rate if CFG["rate_is_usd_per_unit"] else amt / rate
        else:
            df["amount_usd"] = amt
        paid = df[CFG["status_col"]].astype(str).str.lower().eq("paid") if CFG["status_col"] in df else True
        df["refund_usd"] = df["amount_usd"].where(df[CFG["status_col"]].astype(str).str.lower().eq("refunded"), 0.0)
        df["is_paid"] = paid
    DF = df
    lines = [f"rows={len(df)}; rates={RATES}", "columns (dtype | examples):"]
    for col in df.columns:
        s = df[col].dropna()
        ex = s.unique()[:8].tolist() if s.nunique() <= 40 else s.head(3).tolist()
        lines.append(f"- {col} ({df[col].dtype}) | {ex}")
    SCHEMA_TEXT = "\n".join(lines)

# ---------------------------------------------------------------- LLM -> pandas
SYSTEM = """You answer questions about a pandas DataFrame `df` (one row per current order, already deduplicated;
dates already in Asia/Kolkata as `date_ist`, plus `year`,`month`,`day`). `revenue_usd` is USD revenue (paid orders only,
0 otherwise). `amount_usd` is the USD amount for any status. Use ONLY the columns listed. Match names/values exactly as
spelled in the data (case-insensitively for filtering, but return names exactly as in the data).
Reply with ONE python expression (no assignments, no prose, no code fences) using df and pd that evaluates to the final
answer: a number, string, or list. Round money with round(x, 2). Today is {today}. Resolve relative dates from it.
Schema:
{schema}
`refund_usd` is the USD amount of refunded orders (0 otherwise); status 'void' is neither revenue nor a refund. 
Customer ids are in `customer`; `qty` is units sold; `product` and `region` are as spelled in the data."""

def ask_llm(question, error=None):
    msgs = [{"role": "system", "content": SYSTEM.format(today=time.strftime("%Y-%m-%d"), schema=SCHEMA_TEXT)},
            {"role": "user", "content": question}]
    if error:
        msgs.append({"role": "user", "content": f"That expression failed: {error}. Return a corrected single expression."})
    r = httpx.post(f"{LLM_BASE}/chat/completions", timeout=9,
                   headers={"Authorization": f"Bearer {LLM_KEY}"},
                   json={"model": LLM_MODEL, "temperature": 0, "messages": msgs})
    r.raise_for_status()
    t = r.json()["choices"][0]["message"]["content"].strip()
    return re.sub(r"^```(?:python)?|```$", "", t, flags=re.M).strip()

BAD = re.compile(r"__|import|open\(|exec|eval|os\.|sys\.|subprocess|getattr|globals|locals")

def run_expr(expr):
    if BAD.search(expr):
        raise ValueError("unsafe expression")
    return eval(expr, {"__builtins__": {"round": round, "len": len, "sum": sum, "min": min, "max": max,
                                          "sorted": sorted, "abs": abs, "int": int, "float": float, "str": str,
                                          "list": list, "set": set}}, {"df": DF, "pd": pd})

def clean(v):
    if isinstance(v, pd.DataFrame):
        v = v.iloc[:, 0] if v.shape[1] == 1 else v.to_dict("records")
    if isinstance(v, pd.Series):
        v = v.tolist() if len(v) != 1 else v.iloc[0]
    if hasattr(v, "item") and not isinstance(v, (list, dict)):
        v = v.item()
    if isinstance(v, float):
        v = round(v, 2)
    if isinstance(v, list):
        v = [clean(x) for x in v]
    if isinstance(v, (pd.Timestamp,)):
        v = str(v)
    return v

def answer(question):
    if DF is None:
        build()
    err, expr = None, None
    for _ in range(2):
        expr = ask_llm(question, err)
        try:
            return clean(run_expr(expr))
        except Exception as e:  # retry once with the error
            err = f"{expr!r} -> {type(e).__name__}: {e}"
    return None

# ---------------------------------------------------------------- web app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def _startup():
    try:
        build()
    except Exception as e:
        print("startup load failed, will retry on first request:", e)

@app.post("/")
@app.post("/ask")
def ask(body: dict):
    return {"answer": answer(str(body.get("question", "")))}

@app.get("/")
def health():
    return {"ok": True, "rows": None if DF is None else len(DF)}

if __name__ == "__main__" and "--inspect" in sys.argv:
    build()
    print(SCHEMA_TEXT)
    print(DF.head(10).to_string())