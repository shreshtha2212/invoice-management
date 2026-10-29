=import base64
import hashlib
import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# 1. Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Invoice Automation - Review & Edit Suite", layout="wide")
st.title("📄 Enterprise Invoice Extraction & Reconciliation Suite")
st.write(
    "Upload a batch of invoices, extract GST/tax fields, review flagged entries "
    "side-by-side with the source document, and export a reconciled dataset."
)

DEFAULT_MODEL = "gemini-3.8-flash"
MAX_RETRIES = 3
MAX_FILE_MB = 15          # inline upload limit guard
MAX_WORKERS = 4           # parallel extraction threads (keep modest for rate limits)
TOLERANCE = 1.0           # rupee tolerance for reconciliation
MIME_MAP = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


# ----------------------------------------------------------------------------
# 2. Schema
# ----------------------------------------------------------------------------
class InvoiceSchema(BaseModel):
    vendor_name: str = Field(..., description="Legal/trade name of the supplier")
    invoice_number: str = Field(..., description="Invoice number exactly as printed")
    invoice_date: Optional[str] = Field(
        None, description="Invoice issue date in YYYY-MM-DD format. Indian dates are DD/MM/YYYY."
    )
    due_date: Optional[str] = Field(None, description="Payment due date in YYYY-MM-DD format")
    vendor_gstin: Optional[str] = Field(None, description="15-character GSTIN of the supplier, uppercase, no spaces")
    buyer_gstin: Optional[str] = Field(None, description="15-character GSTIN of the recipient, uppercase, no spaces")
    taxable_amount: Optional[float] = Field(None, description="Subtotal before taxes, as a plain number")
    cgst_amount: Optional[float] = Field(None, description="Central GST amount in currency units (not the rate)")
    sgst_amount: Optional[float] = Field(None, description="State GST amount in currency units (not the rate)")
    igst_amount: Optional[float] = Field(None, description="Integrated GST amount in currency units (not the rate)")
    total_amount: float = Field(..., description="Grand total including all taxes, as a plain number")
    currency: Optional[str] = Field("INR", description="ISO currency code")
    hsn_sac_code: Optional[str] = Field(None, description="HSN or SAC code")


SYSTEM_PROMPT = (
    "You are an accounting assistant extracting data from Indian GST invoices. "
    "Return only values visible in the document; use null when a field is absent. "
    "Never guess or compute missing values. Amounts must be plain numbers with no "
    "currency symbols or thousands separators. Dates must be YYYY-MM-DD."
)

FIELDS = list(InvoiceSchema.model_fields.keys())


# ----------------------------------------------------------------------------
# 3. Validation helpers
# ----------------------------------------------------------------------------
GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_GST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin_checksum_ok(gstin: str) -> bool:
    """Validate the 15th check character of a GSTIN."""
    total = 0
    for i, ch in enumerate(gstin[:14]):
        val = _GST_CHARS.index(ch) * (1 if i % 2 == 0 else 2)
        total += val // 36 + val % 36
    return _GST_CHARS[(36 - total % 36) % 36] == gstin[14]


def check_gstin(label: str, gstin: Optional[str]) -> list:
    if not gstin:
        return []
    g = gstin.strip().upper()
    if not GSTIN_RE.match(g):
        return [("error", f"{label} GSTIN '{gstin}' is not in a valid format.")]
    if not gstin_checksum_ok(g):
        return [("warning", f"{label} GSTIN '{gstin}' fails the checksum (possible OCR/typing error).")]
    return []


def validate_record(rec: dict) -> list:
    """Return a list of (level, message) issues for a record."""
    issues = []
    g = lambda k: rec.get(k) or 0.0  # noqa: E731
    taxable, cgst, sgst, igst, total = (g("taxable_amount"), g("cgst_amount"), g("sgst_amount"),
                                        g("igst_amount"), g("total_amount"))

    if total <= 0:
        issues.append(("error", "Grand total is zero or negative."))

    if taxable > 0:
        calc = taxable + cgst + sgst + igst
        if abs(calc - total) > TOLERANCE:
            issues.append(("error", f"Math mismatch: taxable + taxes = {calc:,.2f} but grand total = {total:,.2f} "
                                    f"(diff {calc - total:+,.2f})."))
    else:
        issues.append(("warning", "Taxable amount missing - totals could not be reconciled."))

    if igst > 0 and (cgst > 0 or sgst > 0):
        issues.append(("warning", "Both IGST and CGST/SGST are present; normally only one applies."))
    if abs(cgst - sgst) > 0.01:
        issues.append(("warning", f"CGST ({cgst:,.2f}) and SGST ({sgst:,.2f}) differ; they are usually equal."))

    issues += check_gstin("Vendor", rec.get("vendor_gstin"))
    issues += check_gstin("Buyer", rec.get("buyer_gstin"))

    # Place-of-supply sanity: same state code => CGST+SGST, different => IGST
    vg, bg = (rec.get("vendor_gstin") or "").strip(), (rec.get("buyer_gstin") or "").strip()
    if len(vg) >= 2 and len(bg) >= 2 and vg[:2].isdigit() and bg[:2].isdigit():
        if vg[:2] == bg[:2] and igst > 0:
            issues.append(("warning", "Vendor and buyer are in the same state, but IGST is charged."))
        elif vg[:2] != bg[:2] and (cgst > 0 or sgst > 0):
            issues.append(("warning", "Vendor and buyer are in different states, but CGST/SGST is charged."))

    inv_d = pd.to_datetime(rec.get("invoice_date"), errors="coerce")
    due_d = pd.to_datetime(rec.get("due_date"), errors="coerce")
    if rec.get("invoice_date") and pd.isna(inv_d):
        issues.append(("warning", "Invoice date could not be parsed."))
    if pd.notna(inv_d) and pd.notna(due_d) and due_d < inv_d:
        issues.append(("warning", "Due date is earlier than the invoice date."))
    if not rec.get("invoice_date"):
        issues.append(("warning", "Invoice date is missing."))
    return issues


def find_duplicates(records: list) -> set:
    """IDs of records sharing (vendor GSTIN or name, invoice number) with another record."""
    seen, dupes = {}, set()
    for r in records:
        key = ((r.get("vendor_gstin") or r.get("vendor_name") or "").strip().upper(),
               (r.get("invoice_number") or "").strip().upper())
        if key in seen:
            dupes.update({seen[key], r["id"]})
        else:
            seen[key] = r["id"]
    return dupes


# ----------------------------------------------------------------------------
# 4. Extraction helpers
# ----------------------------------------------------------------------------
def get_mime(filename: str) -> Optional[str]:
    return MIME_MAP.get(os.path.splitext(filename.lower())[1])


def extract_invoice(client, model: str, file_bytes: bytes, mime_type: str) -> InvoiceSchema:
    """Call Gemini with retry + exponential backoff."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                    "Extract the invoice details from this document, including GSTINs, "
                    "tax breakdowns, HSN/SAC codes and dates.",
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=InvoiceSchema,
                ),
            )
            if isinstance(response.parsed, InvoiceSchema):
                return response.parsed
            return InvoiceSchema.model_validate_json(response.text)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    raise last_err


def worker(client, model, item):
    parsed = extract_invoice(client, model, item["bytes"], item["mime"])
    return item, parsed


def get_default_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("GEMINI_API_KEY", "")


# ----------------------------------------------------------------------------
# 5. Sidebar & state
# ----------------------------------------------------------------------------
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Gemini API Key", type="password", value=get_default_key())
st.sidebar.markdown("[Get an API key here](https://aistudio.google.com/)")
model_name = st.sidebar.text_input("Model", value=DEFAULT_MODEL, help="Change this if the model name is unavailable.")
reprocess = st.sidebar.checkbox("Re-process files already extracted", value=False)

st.session_state.setdefault("extracted_records", [])
st.session_state.setdefault("failed_files", [])

if st.sidebar.button("🗑️ Clear all results"):
    st.session_state["extracted_records"] = []
    st.session_state["failed_files"] = []
    st.rerun()

# ----------------------------------------------------------------------------
# 6. Upload & batch processing
# ----------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload Invoice Batch (PDF, PNG, JPG)",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True,
)

if not api_key:
    st.warning("👈 Enter your Gemini API key in the sidebar (or set GEMINI_API_KEY).")
if not uploaded_files:
    st.info("Upload one or more invoices to begin.")

if uploaded_files and api_key:
    done_ids = {r["id"] for r in st.session_state["extracted_records"]}
    queue, skipped = [], []
    for f in uploaded_files:
        data = f.getvalue()
        fid = hashlib.sha1(data).hexdigest()[:12]
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            skipped.append(f"{f.name} (larger than {MAX_FILE_MB} MB)")
        elif fid in done_ids and not reprocess:
            skipped.append(f"{f.name} (already processed)")
        else:
            queue.append({"id": fid, "name": f.name, "bytes": data, "mime": get_mime(f.name)})

    if skipped:
        st.caption("Skipped: " + "; ".join(skipped))

    if queue and st.button(f"🚀 Process {len(queue)} file(s) with AI"):
        client = genai.Client(api_key=api_key)
        progress, status = st.progress(0.0), st.empty()
        results = {r["id"]: r for r in st.session_state["extracted_records"]}
        failed = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(worker, client, model_name, item): item for item in queue}
            for n, fut in enumerate(as_completed(futures), start=1):
                item = futures[fut]
                status.text(f"Processed {n}/{len(queue)}: {item['name']}")
                try:
                    _, parsed = fut.result()
                    rec = parsed.model_dump()
                    rec.update(id=item["id"], source_file=item["name"], file_bytes=item["bytes"],
                               mime=item["mime"], reviewed=False, edited=False)
                    results[item["id"]] = rec
                except Exception as e:  # noqa: BLE001
                    failed.append({"file": item["name"], "error": str(e)})
                progress.progress(n / len(queue))

        st.session_state["extracted_records"] = list(results.values())
        st.session_state["failed_files"] = failed
        status.text("Batch processing complete!")
        st.success(f"Extracted {len(queue) - len(failed)} of {len(queue)} invoices.")

if st.session_state["failed_files"]:
    with st.expander(f"⚠️ {len(st.session_state['failed_files'])} file(s) failed", expanded=True):
        st.dataframe(pd.DataFrame(st.session_state["failed_files"]), use_container_width=True)

# ----------------------------------------------------------------------------
# 7. Review, correction & reconciliation
# ----------------------------------------------------------------------------
records = st.session_state["extracted_records"]

if records:
    dupes = find_duplicates(records)
    issues_by_id = {}
    for r in records:
        iss = validate_record(r)
        if r["id"] in dupes:
            iss.append(("warning", "Possible duplicate: same vendor and invoice number appears more than once."))
        issues_by_id[r["id"]] = iss

    def status_of(r):
        iss = issues_by_id[r["id"]]
        if any(lvl == "error" for lvl, _ in iss):
            return "🔴 Error"
        if iss:
            return "🟠 Warning"
        return "🟢 Clean"

    st.markdown("---")
    st.header("🔍 Review, Correction & Reconciliation Suite")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Invoices", len(records))
    m2.metric("Need attention", sum(1 for r in records if issues_by_id[r["id"]]))
    m3.metric("Reviewed", sum(1 for r in records if r["reviewed"]))
    m4.metric("Total value", f"{sum(r.get('total_amount') or 0 for r in records):,.2f}")

    only_flagged = st.checkbox("Show only invoices with issues", value=False)
    visible = [r for r in records if issues_by_id[r["id"]]] if only_flagged else records
    if not visible:
        st.success("No flagged invoices 🎉")
    else:
        by_id = {r["id"]: r for r in visible}
        selected_id = st.selectbox(
            "Select Invoice to Inspect & Edit",
            options=list(by_id.keys()),
            format_func=lambda i: f"{status_of(by_id[i])}  {by_id[i]['source_file']}  "
                                  f"({by_id[i].get('invoice_number') or 'no number'})"
                                  f"{'  ✔ reviewed' if by_id[i]['reviewed'] else ''}",
        )
        rec = by_id[selected_id]
        rid = rec["id"]

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📄 Original Document")
            if rec["mime"] == "application/pdf":
                if len(rec["file_bytes"]) < 6 * 1024 * 1024:
                    b64 = base64.b64encode(rec["file_bytes"]).decode()
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="720"></iframe>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("PDF too large for inline preview.")
                st.download_button("Download Source PDF", rec["file_bytes"], file_name=rec["source_file"],
                                   mime="application/pdf", key=f"dl_{rid}")
            else:
                st.image(rec["file_bytes"], caption=rec["source_file"], use_container_width=True)

        with col_right:
            st.subheader("✍️ Edit Extracted Fields")

            issues = issues_by_id[rid]
            if not issues:
                st.success("✅ All checks passed (math, GSTIN, dates, duplicates).")
            for level, msg in issues:
                (st.error if level == "error" else st.warning)(msg)

            with st.form(key=f"form_{rid}"):
                v_name = st.text_input("Vendor Name", rec.get("vendor_name") or "", key=f"vn_{rid}")
                c1, c2 = st.columns(2)
                inv_no = c1.text_input("Invoice Number", rec.get("invoice_number") or "", key=f"in_{rid}")
                currency = c2.text_input("Currency", rec.get("currency") or "INR", key=f"cur_{rid}")
                d1, d2 = st.columns(2)
                inv_date = d1.text_input("Invoice Date (YYYY-MM-DD)", rec.get("invoice_date") or "", key=f"id_{rid}")
                due_date = d2.text_input("Due Date (YYYY-MM-DD)", rec.get("due_date") or "", key=f"dd_{rid}")
                g1, g2 = st.columns(2)
                v_gstin = g1.text_input("Vendor GSTIN", rec.get("vendor_gstin") or "", key=f"vg_{rid}")
                b_gstin = g2.text_input("Buyer GSTIN", rec.get("buyer_gstin") or "", key=f"bg_{rid}")

                num = dict(step=0.01, format="%.2f")
                t1, t2 = st.columns(2)
                taxable = t1.number_input("Taxable Amount", value=float(rec.get("taxable_amount") or 0.0),
                                          key=f"ta_{rid}", **num)
                grand = t2.number_input("Grand Total", value=float(rec.get("total_amount") or 0.0),
                                        key=f"gt_{rid}", **num)
                s1, s2, s3 = st.columns(3)
                cgst = s1.number_input("CGST", value=float(rec.get("cgst_amount") or 0.0), key=f"cg_{rid}", **num)
                sgst = s2.number_input("SGST", value=float(rec.get("sgst_amount") or 0.0), key=f"sg_{rid}", **num)
                igst = s3.number_input("IGST", value=float(rec.get("igst_amount") or 0.0), key=f"ig_{rid}", **num)

                hsn = st.text_input("HSN/SAC Code", rec.get("hsn_sac_code") or "", key=f"hs_{rid}")
                reviewed = st.checkbox("Mark as reviewed", value=rec["reviewed"], key=f"rv_{rid}")

                if st.form_submit_button("💾 Save Corrections"):
                    rec.update(
                        vendor_name=v_name.strip(), invoice_number=inv_no.strip(), currency=currency.strip().upper(),
                        invoice_date=inv_date.strip() or None, due_date=due_date.strip() or None,
                        vendor_gstin=v_gstin.strip().upper() or None, buyer_gstin=b_gstin.strip().upper() or None,
                        taxable_amount=taxable, total_amount=grand, cgst_amount=cgst, sgst_amount=sgst,
                        igst_amount=igst, hsn_sac_code=hsn.strip() or None, reviewed=reviewed, edited=True,
                    )
                    st.rerun()  # re-run so the checks above reflect the saved values

    # ------------------------------------------------------------------------
    # 8. Export
    # ------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Master Reconciled Dataset")

    rows = []
    for r in records:
        row = {k: r.get(k) for k in FIELDS}
        row.update(
            source_file=r["source_file"],
            status=status_of(r).split(" ", 1)[1],
            reviewed=r["reviewed"],
            edited=r["edited"],
            issues=" | ".join(m for _, m in issues_by_id[r["id"]]),
        )
        rows.append(row)
    df_master = pd.DataFrame(rows)

    export_scope = st.radio("Export", ["All invoices", "Reviewed only", "Clean only (no issues)"], horizontal=True)
    if export_scope == "Reviewed only":
        df_export = df_master[df_master["reviewed"]]
    elif export_scope == "Clean only (no issues)":
        df_export = df_master[df_master["issues"] == ""]
    else:
        df_export = df_master

    st.dataframe(df_export, use_container_width=True)

    d1, d2 = st.columns(2)
    d1.download_button(
        "📥 Download CSV",
        df_export.to_csv(index=False).encode("utf-8-sig"),  # BOM so Excel opens ₹/unicode correctly
        file_name="master_reconciled_invoices.csv",
        mime="text/csv",
    )
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            df_export.to_excel(xw, index=False, sheet_name="Invoices")
        d2.download_button(
            "📥 Download Excel",
            buf.getvalue(),
            file_name="master_reconciled_invoices.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        d2.caption("Install `openpyxl` to enable Excel export.")
