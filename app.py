import hashlib
import io
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Optional: nicer PDF preview (pip install pymupdf). App works without it.
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")  # pin a model you have tested
MAX_FILE_MB = 15
MAX_RETRIES = 4
MAX_WORKERS = 4
RECON_TOLERANCE = 1.0  # rupees

MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

st.set_page_config(page_title="Invoice Extraction & Reconciliation", layout="wide")


# ----------------------------------------------------------------------------
# Schema (no hard defaults / required totals: the model must be allowed to say "null")
# ----------------------------------------------------------------------------
class LineItem(BaseModel):
    description: Optional[str] = None
    hsn_sac_code: Optional[str] = None
    quantity: Optional[float] = None
    taxable_value: Optional[float] = None
    gst_rate_percent: Optional[float] = None


class InvoiceSchema(BaseModel):
    document_type: Optional[str] = Field(
        None, description="One of: Tax Invoice, Credit Note, Debit Note, Bill of Supply, Other"
    )
    vendor_name: Optional[str] = None
    vendor_gstin: Optional[str] = Field(None, description="15-character GSTIN of the supplier")
    buyer_name: Optional[str] = None
    buyer_gstin: Optional[str] = Field(None, description="15-character GSTIN of the recipient")
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = Field(None, description="Format YYYY-MM-DD")
    due_date: Optional[str] = Field(None, description="Format YYYY-MM-DD")
    place_of_supply: Optional[str] = Field(None, description="State name and/or code as printed")
    reverse_charge: Optional[bool] = Field(None, description="True only if reverse charge is marked Yes")
    taxable_amount: Optional[float] = Field(None, description="Total taxable value before GST")
    cgst_amount: Optional[float] = None
    sgst_amount: Optional[float] = Field(None, description="SGST or UTGST amount")
    igst_amount: Optional[float] = None
    cess_amount: Optional[float] = None
    round_off: Optional[float] = None
    total_amount: Optional[float] = Field(None, description="Grand total payable")
    currency: Optional[str] = Field(None, description="ISO code, e.g. INR")
    hsn_sac_code: Optional[str] = Field(None, description="Primary HSN/SAC if one applies to the whole invoice")
    line_items: Optional[List[LineItem]] = None


PROMPT = """You are extracting data from an Indian GST invoice for an accounting firm.
Rules:
- Copy values exactly as printed. If a field is not clearly present, return null. Never guess or calculate missing values.
- Amounts: plain numbers only, no currency symbols or thousands separators.
- Dates: convert to YYYY-MM-DD.
- GSTIN: 15 characters, uppercase, no spaces. The vendor/supplier is the party issuing the invoice.
- If the document is not an invoice, set document_type to "Other" and leave the rest null.
- Include every line item with its HSN/SAC, taxable value and GST rate if shown."""

HEADER_FIELDS = [
    "document_type", "vendor_name", "vendor_gstin", "buyer_name", "buyer_gstin",
    "invoice_number", "invoice_date", "due_date", "place_of_supply", "reverse_charge",
    "taxable_amount", "cgst_amount", "sgst_amount", "igst_amount", "cess_amount",
    "round_off", "total_amount", "currency", "hsn_sac_code",
]
AMOUNT_FIELDS = ["taxable_amount", "cgst_amount", "sgst_amount", "igst_amount", "cess_amount", "round_off", "total_amount"]


# ----------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------
def _is_transient(err: Exception) -> bool:
    code = getattr(err, "code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    msg = str(err).upper()
    return any(t in msg for t in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE", "TIMEOUT"))


def extract_invoice(api_key: str, model: str, file_name: str, file_bytes: bytes):
    """Returns (data_dict | None, error_message | None). Safe to run in a worker thread."""
    ext = os.path.splitext(file_name.lower())[1]
    mime = MIME_BY_EXT.get(ext)
    if not mime:
        return None, f"Unsupported file type: {ext}"
    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        return None, f"File larger than {MAX_FILE_MB} MB"

    client = genai.Client(api_key=api_key)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=InvoiceSchema,
                    temperature=0,
                ),
            )
            parsed = InvoiceSchema.model_validate_json(resp.text)
            return parsed.model_dump(), None
        except Exception as e:  # retry only on rate-limit / transient server errors
            last_err = e
            if _is_transient(e) and attempt < MAX_RETRIES - 1:
                time.sleep((2 ** attempt) + random.random())
                continue
            break
    return None, f"{type(last_err).__name__}: {last_err}"


# ----------------------------------------------------------------------------
# Validation (this is what makes the tool useful to a CA, not just the extraction)
# ----------------------------------------------------------------------------
GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin_problem(gstin: Optional[str]) -> Optional[str]:
    if not gstin:
        return None
    g = gstin.strip().upper()
    if not GSTIN_RE.match(g):
        return "invalid format"
    state = int(g[:2])
    if not (1 <= state <= 38 or state in (97, 99)):
        return "invalid state code"
    factor, total = 1, 0
    for ch in g[:14]:
        prod = _CHARS.index(ch) * factor
        total += prod // 36 + prod % 36
        factor = 2 if factor == 1 else 1
    if _CHARS[(36 - total % 36) % 36] != g[14]:
        return "checksum failed (likely OCR error)"
    return None


def _parse_date(s: Optional[str]):
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def validate(r: dict) -> List[str]:
    flags = []
    if r.get("error"):
        return [f"Extraction failed: {r['error']}"]
    if r.get("document_type") == "Other":
        flags.append("Not recognised as an invoice")

    for f, label in [("vendor_name", "Vendor name"), ("invoice_number", "Invoice number"),
                     ("invoice_date", "Invoice date"), ("total_amount", "Total amount")]:
        if r.get(f) in (None, "", 0, 0.0):
            flags.append(f"{label} missing")

    for f, label in [("vendor_gstin", "Vendor GSTIN"), ("buyer_gstin", "Buyer GSTIN")]:
        p = gstin_problem(r.get(f))
        if p:
            flags.append(f"{label}: {p}")

    inv_date, due_date = _parse_date(r.get("invoice_date")), _parse_date(r.get("due_date"))
    if r.get("invoice_date") and not inv_date:
        flags.append("Invoice date not parseable")
    if inv_date and due_date and due_date < inv_date:
        flags.append("Due date is before invoice date")

    cgst, sgst, igst = (r.get(k) or 0.0 for k in ("cgst_amount", "sgst_amount", "igst_amount"))
    if abs(cgst - sgst) > 0.01:
        flags.append("CGST and SGST differ (should normally be equal)")
    if igst and (cgst or sgst):
        flags.append("Both IGST and CGST/SGST present")

    # Intra-state vs inter-state check
    v_state = (r.get("vendor_gstin") or "")[:2]
    pos = re.match(r"\s*(\d{2})", r.get("place_of_supply") or "")
    dest_state = pos.group(1) if pos else (r.get("buyer_gstin") or "")[:2]
    if v_state.isdigit() and dest_state.isdigit():
        if v_state == dest_state and igst:
            flags.append("Same-state supply but IGST charged")
        if v_state != dest_state and (cgst or sgst):
            flags.append("Inter-state supply but CGST/SGST charged")

    # Total reconciliation
    taxable = r.get("taxable_amount")
    total = r.get("total_amount")
    if taxable is None:
        flags.append("Taxable amount missing, cannot reconcile")
    elif total is not None:
        calc = taxable + cgst + sgst + igst + (r.get("cess_amount") or 0.0) + (r.get("round_off") or 0.0)
        if abs(calc - total) > RECON_TOLERANCE:
            flags.append(f"Total mismatch: components = {calc:,.2f}, invoice total = {total:,.2f}")
    return flags


def duplicate_ids(records: dict) -> dict:
    """Maps record id -> id of the first record with same vendor + invoice number."""
    seen, dups = {}, {}
    for rid, r in records.items():
        if r.get("error") or not r.get("invoice_number"):
            continue
        key = ((r.get("vendor_gstin") or r.get("vendor_name") or "").strip().lower(),
               str(r["invoice_number"]).strip().lower())
        if key in seen:
            dups[rid] = seen[key]
        else:
            seen[key] = rid
    return dups


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_api_key() -> str:
    try:
        secret = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret = ""
    return secret or os.getenv("GEMINI_API_KEY", "")


@st.cache_data(show_spinner=False)
def render_pdf_pages(pdf_bytes: bytes, max_pages: int = 3):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return [doc[i].get_pixmap(dpi=110).tobytes("png") for i in range(min(len(doc), max_pages))]


def all_flags(records: dict) -> dict:
    dups = duplicate_ids(records)
    out = {}
    for rid, r in records.items():
        f = validate(r)
        if rid in dups:
            f.append(f"Possible duplicate of {records[dups[rid]]['source_file']}")
        out[rid] = f
    return out


def build_excel(records: dict, flags: dict, only_reviewed: bool) -> bytes:
    rows, items = [], []
    for rid, r in records.items():
        if r.get("error") or (only_reviewed and not r.get("reviewed")):
            continue
        row = {k: r.get(k) for k in HEADER_FIELDS}
        row.update(source_file=r["source_file"], reviewed=bool(r.get("reviewed")),
                   validation_flags="; ".join(flags[rid]))
        rows.append(row)
        for li in r.get("line_items") or []:
            items.append({"source_file": r["source_file"], "invoice_number": r.get("invoice_number"), **li})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="Invoices", index=False)
        pd.DataFrame(items).to_excel(xw, sheet_name="Line Items", index=False)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("📄 Invoice Extraction & Reconciliation")
st.caption("Upload invoices, let AI extract GST fields, then review flagged items before exporting.")

if "records" not in st.session_state:
    st.session_state["records"] = {}  # file-hash -> record (insertion ordered)
records = st.session_state["records"]

with st.sidebar:
    st.header("Configuration")
    api_key = get_api_key()
    if api_key:
        st.success("API key loaded from server configuration.")
    else:
        api_key = st.text_input("Gemini API key", type="password")
        st.caption("For deployment, set GEMINI_API_KEY in st.secrets or the environment instead.")
    model = st.text_input("Model", value=DEFAULT_MODEL, help="Pin a specific model you have tested.")
    workers = st.slider("Parallel requests", 1, 8, MAX_WORKERS)
    st.info("Invoices contain client financial data. Use a paid/enterprise API tier whose terms exclude "
            "training on your data, and tell clients where their data is processed.")

uploaded = st.file_uploader("Upload invoices (PDF, PNG, JPG)", type=["pdf", "png", "jpg", "jpeg"],
                            accept_multiple_files=True)

if uploaded:
    new_files = []
    for f in uploaded:
        data = f.getvalue()
        rid = hashlib.sha256(data).hexdigest()[:16]
        if rid not in records or records[rid].get("error"):  # skip already-extracted files, retry failed ones
            new_files.append((rid, f.name, data))

    st.write(f"{len(uploaded)} file(s) uploaded, {len(new_files)} to process.")
    if new_files and st.button("🚀 Process with AI", type="primary", disabled=not api_key):
        progress, status = st.progress(0.0), st.empty()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(extract_invoice, api_key, model, name, data): (rid, name, data)
                       for rid, name, data in new_files}
            for n, fut in enumerate(as_completed(futures), 1):
                rid, name, data = futures[fut]
                result, err = fut.result()
                record = {"source_file": name, "file_bytes": data, "error": err, "reviewed": False}
                record.update(result or {})
                records[rid] = record
                progress.progress(n / len(new_files))
                status.text(f"Processed {n}/{len(new_files)}: {name}")
        status.text("Batch complete.")
        st.rerun()
    if not api_key:
        st.warning("Enter a Gemini API key in the sidebar to begin.")

if records:
    flags = all_flags(records)
    n_failed = sum(1 for r in records.values() if r.get("error"))
    n_flagged = sum(1 for rid, f in flags.items() if f and not records[rid].get("error"))
    n_reviewed = sum(1 for r in records.values() if r.get("reviewed"))

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Invoices", len(records))
    c2.metric("Need attention", n_flagged)
    c3.metric("Failed", n_failed)
    c4.metric("Reviewed", n_reviewed)

    def label(rid):
        r = records[rid]
        icon = "❌" if r.get("error") else ("✅" if r.get("reviewed") else ("⚠️" if flags[rid] else "🟢"))
        return f"{icon} {r['source_file']}"

    rid = st.selectbox("Select invoice", list(records.keys()), format_func=label)
    rec = records[rid]
    left, right = st.columns(2)

    with left:
        st.subheader("Original document")
        if rec["source_file"].lower().endswith(".pdf"):
            if fitz:
                for img in render_pdf_pages(rec["file_bytes"]):
                    st.image(img)
            else:
                st.info("Install pymupdf for inline PDF preview.")
            st.download_button("Download source PDF", rec["file_bytes"], file_name=rec["source_file"])
        else:
            st.image(rec["file_bytes"], caption=rec["source_file"])

    with right:
        st.subheader("Extracted fields")
        for f in flags[rid]:
            st.warning(f)
        if not flags[rid]:
            st.success("All automated checks passed.")

        with st.form(key=f"form_{rid}"):
            new_vals = {}
            for field in HEADER_FIELDS:
                cur = rec.get(field)
                key = f"{rid}_{field}"
                if field in AMOUNT_FIELDS:
                    new_vals[field] = st.number_input(field.replace("_", " ").title(),
                                                      value=float(cur or 0.0), step=0.01,
                                                      format="%.2f", key=key)
                elif field == "reverse_charge":
                    new_vals[field] = st.checkbox("Reverse charge", value=bool(cur), key=key)
                else:
                    new_vals[field] = st.text_input(field.replace("_", " ").title(),
                                                    value=cur or "", key=key).strip() or None
            reviewed = st.checkbox("Mark as reviewed", value=bool(rec.get("reviewed")), key=f"{rid}_reviewed")
            if st.form_submit_button("Save"):
                rec.update(new_vals)
                rec["reviewed"] = reviewed
                rec["error"] = None
                st.rerun()

        if rec.get("line_items"):
            with st.expander(f"Line items ({len(rec['line_items'])})"):
                st.dataframe(pd.DataFrame(rec["line_items"]), hide_index=True)

    st.markdown("---")
    st.subheader("Export")
    only_reviewed = st.checkbox("Export reviewed invoices only", value=False)
    export_rows = []
    for k, r in records.items():
        if r.get("error") or (only_reviewed and not r.get("reviewed")):
            continue
        row = {f: r.get(f) for f in HEADER_FIELDS}
        row.update(source_file=r["source_file"], reviewed=bool(r.get("reviewed")),
                   validation_flags="; ".join(flags[k]))
        export_rows.append(row)
    df = pd.DataFrame(export_rows)
    st.dataframe(df, hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("📥 Download Excel (invoices + line items)",
                       build_excel(records, flags, only_reviewed),
                       file_name="reconciled_invoices.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="reconciled_invoices.csv", mime="text/csv")

    if st.button("Clear all"):
        st.session_state["records"] = {}
        st.rerun()
