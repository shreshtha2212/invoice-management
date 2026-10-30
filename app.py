import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional

# 1. Page Configuration
st.set_page_config(page_title="Invoice Automation - Review & Edit Suite", layout="wide")
st.title("📄 Enterprise Invoice Extraction & Reconciliation Suite")
st.write("Upload a batch of invoices, extract comprehensive GST/tax fields, and review/correct entries before exporting.")

# 2. Priority 1: Expanded Schema for Indian Tax & Bookkeeping
class InvoiceSchema(BaseModel):
    vendor_name: str
    invoice_number: str
    invoice_date: Optional[str] = Field(None, description="Date the invoice was issued")
    due_date: Optional[str] = Field(None, description="Payment due date")
    vendor_gstin: Optional[str] = Field(None, description="GSTIN of the supplier/vendor")
    buyer_gstin: Optional[str] = Field(None, description="GSTIN of the recipient/buyer")
    taxable_amount: Optional[float] = Field(None, description="Subtotal amount before taxes")
    cgst_amount: Optional[float] = Field(None, description="Central GST amount")
    sgst_amount: Optional[float] = Field(None, description="State GST amount")
    igst_amount: Optional[float] = Field(None, description="Integrated GST amount")
    total_amount: float = Field(..., description="Grand total including all taxes")
    currency: Optional[str] = Field("INR", description="Currency code")
    hsn_sac_code: Optional[str] = Field(None, description="HSN or SAC code")

# 3. Sidebar for API Key & Demo Mode
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
st.sidebar.markdown("[Get an API key here](https://aistudio.google.com/)")

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Presentation Mode")
enable_demo_mode = st.sidebar.checkbox("Enable Demo / Sample Data (No API Key Required)")

# Initialize session state to store batch results for review
if "extracted_records" not in st.session_state:
    st.session_state["extracted_records"] = []

# Handle Demo Mode Data Injection
if enable_demo_mode and not st.session_state["extracted_records"]:
    st.session_state["extracted_records"] = [
        {
            "vendor_name": "Apex Tech Solutions Pvt Ltd",
            "invoice_number": "INV-2026-0001",
            "invoice_date": "2026-10-15",
            "due_date": "2026-10-30",
            "vendor_gstin": "29AABCA1234F1Z5",
            "buyer_gstin": "29XYZAB9876G1Z2",
            "taxable_amount": 10000.0,
            "cgst_amount": 900.0,
            "sgst_amount": 900.0,
            "igst_amount": 0.0,
            "total_amount": 11800.0,
            "currency": "INR",
            "hsn_sac_code": "998314",
            "source_file": "sample_invoice_1.pdf",
            "file_bytes": b"mock_pdf_bytes"
        },
        {
            "vendor_name": "Bengaluru Office Supplies",
            "invoice_number": "BOS/2026/489",
            "invoice_date": "2026-10-18",
            "due_date": "2026-11-02",
            "vendor_gstin": "29GGGGG5678H1ZU",
            "buyer_gstin": "29XYZAB9876G1Z2",
            "taxable_amount": 5000.0,
            "cgst_amount": 0.0,
            "sgst_amount": 0.0,
            "igst_amount": 900.0,
            "total_amount": 5900.0,
            "currency": "INR",
            "hsn_sac_code": "4820",
            "source_file": "sample_invoice_2.jpg",
            "file_bytes": b"mock_jpg_bytes"
        }
    ]

# 4. Multi-File Uploader Component
uploaded_files = st.file_uploader(
    "Upload Invoice Batch (PDF, PNG, JPG)", 
    type=["png", "jpg", "jpeg", "pdf"], 
    accept_multiple_files=True
)

if uploaded_files and api_key and not enable_demo_mode:
    if st.button("🚀 Process Batch with AI"):
        try:
            client = genai.Client(api_key=api_key)
            records = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_files = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing ({i+1}/{total_files}): {uploaded_file.name}...")
                
                # Accurate Mime Type Detection for PDF, PNG, JPG/JPEG
                file_name_lower = uploaded_file.name.lower()
                if file_name_lower.endswith(".pdf"):
                    mime_type = "application/pdf"
                elif file_name_lower.endswith(".jpg") or file_name_lower.endswith(".jpeg"):
                    mime_type = "image/jpeg"
                else:
                    mime_type = "image/png"
                    
                file_bytes = uploaded_file.read()
                
                try:
                    response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=[
                            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                            (
                                "Extract comprehensive invoice details from this document accurately. "
                                "Ensure taxable amounts, individual tax breakdowns (CGST, SGST, IGST), "
                                "and the grand total align correctly with the document totals."
                            )
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=InvoiceSchema,
                        ),
                    )
                    
                    parsed = InvoiceSchema.model_validate_json(response.text)
                    data_dict = parsed.model_dump()
                    data_dict["source_file"] = uploaded_file.name
                    data_dict["file_bytes"] = file_bytes
                    records.append(data_dict)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / total_files)
                
            st.session_state["extracted_records"] = records
            status_text.text("Batch processing complete!")
            st.success(f"Successfully extracted {len(records)} invoices.")
            
        except Exception as e:
            st.error(f"Initialization or API error: {e}")

# 5. Priority 2: Review, Correction & Reconciliation Suite
if st.session_state["extracted_records"]:
    st.markdown("---")
    st.header("🔍 Review, Correction & Reconciliation Suite")
    st.write("Select an invoice below to inspect its original document side-by-side, verify math reconciliation, and make corrections before final export.")

    if enable_demo_mode:
        st.info("ℹ️ **Demo Mode Active:** Displaying pre-loaded sample invoices to test the review and export workflow without using API credits.")

    file_names = [r["source_file"] for r in st.session_state["extracted_records"]]
    selected_file = st.selectbox("Select Invoice to Inspect & Edit", file_names)

    selected_record = next(r for r in st.session_state["extracted_records"] if r["source_file"] == selected_file)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📄 Original Document Preview")
        if enable_demo_mode:
            st.success("Sample preview loaded for presentation.")
        elif selected_file.lower().endswith(".pdf"):
            st.info("PDF document preview loaded.")
            st.download_button("Download Source PDF", selected_record["file_bytes"], file_name=selected_file)
        else:
            st.image(selected_record["file_bytes"], caption=selected_file, use_container_width=True)

    with col_right:
        st.subheader("✍️ Edit Extracted Fields")
        
        taxable = selected_record.get("taxable_amount") or 0.0
        cgst = selected_record.get("cgst_amount") or 0.0
        sgst = selected_record.get("sgst_amount") or 0.0
        igst = selected_record.get("igst_amount") or 0.0
        grand_total = selected_record.get("total_amount") or 0.0
        
        calculated_total = taxable + cgst + sgst + igst
        difference = abs(calculated_total - grand_total)

        if taxable > 0 and difference > 5.0:
            st.warning(f"⚠️ Potential Mismatch: Calculated Total ({calculated_total:.2f}) differs from Grand Total ({grand_total:.2f}) by {difference:.2f}. (Check for shipping charges or rounding)")
        else:
            st.success("✅ Totals Reconciled Successfully.")

        # Editable Form
        with st.form(key=f"form_{selected_file}"):
            updated_vendor = st.text_input("Vendor Name", value=selected_record.get("vendor_name", ""))
            updated_inv_num = st.text_input("Invoice Number", value=selected_record.get("invoice_number", ""))
            updated_inv_date = st.text_input("Invoice Date", value=selected_record.get("invoice_date") or "")
            updated_due_date = st.text_input("Due Date", value=selected_record.get("due_date") or "")
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                updated_v_gstin = st.text_input("Vendor GSTIN", value=selected_record.get("vendor_gstin") or "")
            with col_g2:
                updated_b_gstin = st.text_input("Buyer GSTIN", value=selected_record.get("buyer_gstin") or "")
                
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                updated_taxable = st.number_input("Taxable Amount", value=float(selected_record.get("taxable_amount") or 0.0))
            with col_t2:
                updated_grand = st.number_input("Grand Total", value=float(selected_record.get("total_amount") or 0.0))
                
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                updated_cgst = st.number_input("CGST", value=float(selected_record.get("cgst_amount") or 0.0))
            with col_s2:
                updated_sgst = st.number_input("SGST", value=float(selected_record.get("sgst_amount") or 0.0))
            with col_s3:
                updated_igst = st.number_input("IGST", value=float(selected_record.get("igst_amount") or 0.0))
                
            updated_hsn = st.text_input("HSN/SAC Code", value=selected_record.get("hsn_sac_code") or "")
            
            submitted = st.form_submit_button("Save Corrections")
            if submitted:
                selected_record["vendor_name"] = updated_vendor
                selected_record["invoice_number"] = updated_inv_num
                selected_record["invoice_date"] = updated_inv_date
                selected_record["due_date"] = updated_due_date
                selected_record["vendor_gstin"] = updated_v_gstin
                selected_record["buyer_gstin"] = updated_b_gstin
                selected_record["taxable_amount"] = updated_taxable
                selected_record["total_amount"] = updated_grand
                selected_record["cgst_amount"] = updated_cgst
                selected_record["sgst_amount"] = updated_sgst
                selected_record["igst_amount"] = updated_igst
                selected_record["hsn_sac_code"] = updated_hsn
                st.success("Changes saved successfully for this invoice!")

    st.markdown("---")
    st.subheader("📊 Master Reconciled Dataset Export")
    
    export_data = []
    for r in st.session_state["extracted_records"]:
        clean_r = {k: v for k, v in r.items() if k not in ["file_bytes", "source_file"]}
        clean_r["source_file"] = r["source_file"]
        export_data.append(clean_r)
        
    df_master = pd.DataFrame(export_data)
    st.dataframe(df_master)
    
    csv_master = df_master.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Master Reconciled CSV",
        data=csv_master,
        file_name="master_reconciled_invoices.csv",
        mime="text/csv",
    )

elif not uploaded_files and not api_key and not enable_demo_mode:
    st.warning("👈 Please enter your Gemini API key in the sidebar, or check **'Enable Demo Mode'** to test the app instantly without an API key.")
