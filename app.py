# app.py
"""
Streamlit Multi-Tab Contract Simplifier & Bias Detector

Usage:
    streamlit run app.py

Requirements:
    pip install streamlit transformers torch pdfplumber nltk pandas sentencepiece
    python -c "import nltk; nltk.download('punkt')"

This app:
 - Tab 1: Upload & Simplify (whole-document one-shot simplify + fair rewrite excerpt)
 - Tab 2: Bias Detector (paragraph-wise clause scan, labels + reasons)
 - Tab 3: Download Reports (CSV download of last run)
"""
from typing import List, Dict, Any, Optional
import streamlit as st
import pdfplumber, io, os, re, time, traceback
import pandas as pd
import html
import nltk

# ensure pun
from nltk.tokenize import sent_tokenize

# Ensure required models exist (runs only first time)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


# --------- Configurable model names (change if you want different models) ----------
ZSC_MODEL = "facebook/bart-large-mnli"            # zero-shot classifier (NLI)
SUMMARIZER_MODEL = "philschmid/distilbart-cnn-12-6-samsum"  # summarizer (light-weight)
# You may change SUMMARIZER_MODEL to "google/flan-t5-base" or other seq2seq models if GPU available.

# Bias keywords used in heuristic scoring
BIAS_KEYWORDS = [
    "sole discretion", "at its discretion", "without notice", "indemnify", "irrevocable",
    "perpetual", "forever", "not liable", "without liability", "no refund", "non-refundable",
    "no recourse", "exclusive", "only the", "anytime", "at will", "terminate for any reason",
    "binding on", "assigns and successors", "waive", "waives", "waiver", "no appeal",
    "court waived", "arbitration", "no jury", "liquidated damages"
]

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Contract Simplifier & Bias Detector", layout="wide")

# ---------- Cached model loaders ----------
@st.cache_resource(show_spinner=False)
def load_zsc_model():
    try:
        from transformers import pipeline
        device = 0 if ( (hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available()) ) else -1
        return pipeline("zero-shot-classification", model=ZSC_MODEL, device=device)
    except Exception as e:
        st.warning(f"Zero-shot model load failed: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_summarizer_model():
    try:
        from transformers import pipeline
        device = 0 if ( (hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available()) ) else -1
        return pipeline("summarization", model=SUMMARIZER_MODEL, device=device)
    except Exception as e:
        st.warning(f"Summarizer model load failed: {e}")
        return None

zsc_pipeline = load_zsc_model()
summarizer = load_summarizer_model()

# ---------- Utilities ----------
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return "\n\n".join(pages)

def simple_clean(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    # keep paragraph separators (converted from double spaces)
    text = text.replace("  ", "\n\n")
    return text.strip()

def split_into_paragraphs(text: str, min_chars: int = 120) -> List[str]:
    text = text.replace("\r", "\n")
    # numbered clause detection
    numbered = re.split(r'(?m)(?=^\s*(\d+(\.\d+)*\.)\s+)', text)
    if len(numbered) > 1:
        chunks = []
        for i in range(0, len(numbered), 2):
            heading = numbered[i].strip()
            body = numbered[i+1].strip() if i+1 < len(numbered) else ""
            chunk = (heading + " " + body).strip()
            if len(chunk) > 20:
                chunks.append(chunk)
        if chunks:
            return chunks
    # paragraph split
    if "\n\n" in text:
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
        if paras:
            return paras
    # fallback: sentence-based grouping
    from nltk.tokenize import sent_tokenize
    sents = sent_tokenize(text)
    chunks = []
    buf = []
    buf_len = 0
    for s in sents:
        buf.append(s)
        buf_len += len(s)
        if buf_len >= min_chars:
            chunks.append(" ".join(buf).strip())
            buf = []
            buf_len = 0
    if buf:
        chunks.append(" ".join(buf).strip())
    return chunks

def contains_keyword(text: str, kw: str) -> bool:
    return re.search(r"\b" + re.escape(kw) + r"\b", text, flags=re.IGNORECASE) is not None

# ---------- Bias detection (hybrid) ----------
def detect_bias(text: str, use_zsc: bool = True) -> Dict[str, Any]:
    reasons = []
    matched = []
    text_short = text if len(text) < 2000 else text[:2000]

    # keyword hits
    kw_hits = 0
    for kw in BIAS_KEYWORDS:
        if contains_keyword(text_short, kw):
            kw_hits += 1
            matched.append(kw)
            reasons.append(f"Contains phrase: '{kw}'")
    kw_score = min(0.15 * kw_hits, 0.6)

    zsc_score = 0.0
    zsc_label = None
    if use_zsc and (zsc_pipeline is not None):
        try:
            candidate_labels = ["unfair", "one-sided", "balanced", "neutral", "exploitative"]
            z = zsc_pipeline(text_short, candidate_labels)
            label_scores = {lab: sc for lab, sc in zip(z['labels'], z['scores'])}
            unfair_like = label_scores.get("unfair", 0.0) + label_scores.get("one-sided", 0.0) + label_scores.get("exploitative", 0.0)
            zsc_score = float(unfair_like) * 0.6
            zsc_label = z['labels'][0] if z['labels'] else None
            if unfair_like > 0.25:
                reasons.append(f"Model signals possible unfairness (top: {', '.join(z['labels'][:3])})")
        except Exception:
            zsc_score = 0.0

    one_sided_bonus = 0.0
    if contains_keyword(text_short, "only the") or contains_keyword(text_short, "exclusive"):
        one_sided_bonus = 0.15
        reasons.append("Exclusive/only-language detected")

    raw_score = kw_score + zsc_score + one_sided_bonus
    score = float(max(0.0, min(1.0, raw_score)))

    if score < 0.25:
        label = "Low"
    elif score < 0.55:
        label = "Moderate"
    else:
        label = "High"

    if not reasons and score > 0.05:
        reasons.append("Heuristic signals detected")

    return {"score": round(score, 3), "label": label, "reasons": reasons, "matched_keywords": list(set(matched)), "zsc_top_label": zsc_label}

# ---------- Simplification using summarizer pipeline ----------
def flan_simplify_whole(text: str, style: str = "both", max_new_tokens: int = 256) -> str:
    # Single-shot summarization of whole contract (with head+tail if too long)
    txt = text.strip()
    if len(txt) > 8000:
        head = txt[:4000]
        tail = txt[-4000:]
        txt = head + "\n\n" + tail
    if summarizer is not None:
        try:
            out = summarizer(txt, max_length=max_new_tokens, min_length=30, do_sample=False)
            summary = out[0].get("summary_text", "") if isinstance(out, list) else str(out)
        except Exception as e:
            summary = txt
    else:
        # fallback: naive extraction
        summary = txt if len(txt) < 2000 else txt[:2000] + " ..."
    if style == "plain":
        return summary
    bullets = "- " + summary.replace(". ", ".\n- ")
    if style == "bullet":
        return bullets
    return f"Plain: {summary}\n\nBullet Points:\n{bullets}"

def suggest_fair_version_simple(text: str) -> str:
    s = text
    s = re.sub(r"\bat (its|their) sole discretion\b", "after mutual agreement or with reasonable notice", s, flags=re.IGNORECASE)
    s = re.sub(r"\bwithout notice\b", "with reasonable prior notice", s, flags=re.IGNORECASE)
    s = re.sub(r"\bnot liable for (.+?)\.", "liable for such matters within reasonable limits.", s, flags=re.IGNORECASE)
    return s

# ---------- App state ----------
if "last_df" not in st.session_state:
    st.session_state["last_df"] = pd.DataFrame()

# ---------- Layout: Multi-Tab ----------
st.title("Contract Simplifier & Bias Detector — Multi-Tab")

tabs = st.tabs(["Upload & Simplify", "Bias Detector", "Download Reports"])

# ---------------- Tab 1: Upload & Simplify ----------------
with tabs[0]:
    st.header("Upload & Simplify (single-shot)")
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded = st.file_uploader("Upload a contract (PDF or TXT)", type=["pdf", "txt"])
        paste = st.text_area("Or paste contract text here (optional)", height=180)
        simplify_style = st.selectbox("Simplification style", options=["both", "plain", "bullet"], index=0)
        run_quick = st.button("Quick Simplify + Fair Rewrite")
    with col2:
        st.markdown("**Quick outputs (one-shot)**")
        quick_out = st.empty()
        side_out = st.empty()

    if run_quick:
        raw_text = ""
        if uploaded is not None:
            try:
                b = uploaded.read()
                if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
                    raw_text = extract_text_from_pdf_bytes(b)
                else:
                    raw_text = b.decode("utf-8", errors="ignore")
            except Exception as e:
                st.error(f"Failed to read file: {e}")
                raw_text = ""
        else:
            raw_text = paste or ""
        raw_text = simple_clean(raw_text)
        if not raw_text:
            st.warning("Please upload a PDF/TXT or paste some contract text.")
        else:
            t0 = time.time()
            simplified = flan_simplify_whole(raw_text, style=simplify_style)
            fair = suggest_fair_version_simple(raw_text)
            bias = detect_bias(raw_text, use_zsc=True)
            elapsed = time.time() - t0

            # show outputs
            plain = simplified.split("\n\n")[0] if "\n\n" in simplified else simplified
            bullets = simplified.split("\n\n")[1] if "\n\n" in simplified else ""
            quick_html = f"""
            <div style='font-family:Arial,Helvetica,sans-serif;'>
              <div style='padding:8px;border-radius:6px;border:1px solid #eee;background:#fafafa;margin-bottom:8px;'>
                <strong>Overall Bias:</strong> {bias['label']} (score: {bias['score']})
              </div>
              <div style='font-weight:700;'>Simplified — Plain</div>
              <div style='padding:8px;border:1px solid #eee;border-radius:6px;background:#fff;white-space:pre-wrap;margin-bottom:8px;'>{html.escape(plain)}</div>
              <div style='font-weight:700;'>Simplified — Bullets</div>
              <div style='padding:8px;border:1px solid #eee;border-radius:6px;background:#fff;white-space:pre-wrap;'>{html.escape(bullets)}</div>
              <div style='margin-top:10px;color:#666;font-size:12px;'>Processed in {elapsed:.2f}s</div>
            </div>
            """
            quick_out.markdown(quick_html, unsafe_allow_html=True)

            side_html = f"""
            <div style='display:flex;gap:12px;flex-wrap:wrap;'>
              <div style='flex:1;min-width:280px;'>
                <div style='font-weight:700;margin-bottom:6px;'>Original (excerpt)</div>
                <div style='padding:8px;border:1px solid #eee;border-radius:6px;background:#fff;white-space:pre-wrap;'>{html.escape(raw_text[:4000])}</div>
              </div>
              <div style='flex:1;min-width:280px;'>
                <div style='font-weight:700;margin-bottom:6px;'>Fair Rewritten (excerpt)</div>
                <div style='padding:8px;border:1px solid #eee;border-radius:6px;background:#fff;white-space:pre-wrap;'>{html.escape(fair[:4000])}</div>
              </div>
            </div>
            """
            side_out.markdown(side_html, unsafe_allow_html=True)

            # save quick result to session (for download tab)
            st.session_state["last_df"] = pd.DataFrame([{
                "Original (excerpt)": raw_text[:10000],
                "Simplified": simplified,
                "Fair Rewritten (excerpt)": fair,
                "Bias Label": bias["label"],
                "Bias Score": bias["score"]
            }])

# ---------------- Tab 2: Bias Detector ----------------
with tabs[1]:
    st.header("Bias Detector — Clause-level (paragraph-wise)")
    colA, colB = st.columns([1, 2])
    with colA:
        uploaded2 = st.file_uploader("Upload contract for clause scan (PDF/TXT)", key="u2", type=["pdf", "txt"])
        paste2 = st.text_area("Or paste contract text here", key="p2", height=200)
        sensitivity = st.selectbox("Sensitivity", options=["Mild", "Moderate", "Strict"], index=1)
        show_jargon = st.checkbox("Highlight matched keywords in results", value=True)
        run_deep = st.button("Run Deep Clause Scan")
    with colB:
        scan_out = st.empty()

    if run_deep:
        raw_text = ""
        if uploaded2 is not None:
            try:
                b = uploaded2.read()
                if uploaded2.type == "application/pdf" or uploaded2.name.lower().endswith(".pdf"):
                    raw_text = extract_text_from_pdf_bytes(b)
                else:
                    raw_text = b.decode("utf-8", errors="ignore")
            except Exception as e:
                st.error(f"Failed to read file: {e}")
                raw_text = ""
        else:
            raw_text = paste2 or ""
        raw_text = simple_clean(raw_text)
        if not raw_text:
            st.warning("Please upload text or paste contract.")
        else:
            paras = split_into_paragraphs(raw_text, min_chars=120)
            rows = []
            t0 = time.time()
            for idx, p in enumerate(paras, start=1):
                if len(p.strip()) < 30:
                    continue
                bias = detect_bias(p, use_zsc=True)
                # sensitivity adjust
                score = bias["score"]
                if sensitivity == "Mild":
                    score = score * 0.7
                elif sensitivity == "Strict":
                    score = min(1.0, score * 1.25)
                if score < 0.25:
                    label = "Low"
                elif score < 0.55:
                    label = "Moderate"
                else:
                    label = "High"
                rows.append({
                    "Clause #": idx,
                    "Original": p,
                    "Bias Label": label,
                    "Bias Score": round(score, 3),
                    "Matched Keywords": ", ".join(bias.get("matched_keywords", [])),
                    "Bias Reasons": "; ".join(bias.get("reasons", []))
                })
            t1 = time.time()
            df_scan = pd.DataFrame(rows)
            if df_scan.empty:
                scan_out.info("No clauses found or nothing to analyze.")
            else:
                # show table
                def color_label(val):
                    if val == "High":
                        return "background-color:#ffe6e6"
                    elif val == "Moderate":
                        return "background-color:#fff8dc"
                    else:
                        return "background-color:#e9f7ec"

                styled = df_scan.style.applymap(lambda v: color_label(v) if v in ["High","Moderate","Low"] else "", subset=["Bias Label"])
                st.write(f"Processed {len(df_scan)} clauses in {t1-t0:.1f}s")
                st.dataframe(df_scan, use_container_width=True)
                # store for download
                st.session_state["last_df"] = df_scan

                if show_jargon:
                    # show keyword frequency
                    kw_hits = {}
                    for kw in BIAS_KEYWORDS:
                        cnt = df_scan["Original"].str.lower().str.contains(re.escape(kw)).sum()
                        if cnt > 0:
                            kw_hits[kw] = int(cnt)
                    if kw_hits:
                        st.markdown("**Detected keyword hits (clause counts):**")
                        st.write(pd.Series(kw_hits).sort_values(ascending=False))

# ---------------- Tab 3: Download Reports ----------------
        if 'df_last' in globals() and hasattr(df_last, "columns") and "Simplified" in df_last.columns:

            txt_out = ""
            for _, r in df_last.iterrows():
                original = r.get("Original (excerpt)", "")
                simplified = r.get("Simplified", "")
                fair = r.get("Fair Rewritten (excerpt)", "")
                bias_label = r.get("Bias Label", "")
                bias_score = r.get("Bias Score", "")

                txt_out += (
                    f"Original:\n{original}\n\n"
                    f"Simplified:\n{simplified}\n\n"
                    f"Fair Rewrite:\n{fair}\n\n"
                    f"Bias: {bias_label} (score: {bias_score})\n"
                    f"----------------------------\n\n"
                )

            st.download_button(
                "Download TXT (Simplified + Fair Rewrite)",
                data=txt_out,
                file_name="contract_simplified.txt",
                mime="text/plain"
            )
        else:
            ...
st.markdown("---")
st.caption("This tool provides educational simplifications and suggestions only. It is not legal advice.")
