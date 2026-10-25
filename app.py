import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. Page Configuration
st.set_page_config(page_title="Bulk Invoice Automation", layout="centered")
st.title("📄 Bulk Invoice Automation Tool")
st.write("Drag and drop your batch of mixed invoices (PDFs, PNGs, JPEGs) below.")

# 2. Schema Definition
class InvoiceSchema(BaseModel):
    vendor_name: str
    invoice_number: str
    total_amount: float

# 3. Sidebar for Client's API Key
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
st.sidebar.markdown("[Get an API key here](https://aistudio.google.com/)")

# 4. Multi-File Uploader Component
uploaded_files = st.file_uploader(
    "Upload Invoice Batch", 
    type=["png", "jpg", "jpeg", "pdf"], 
    accept_multiple_files=True
)

if uploaded_files and api_key:
    if st.button(f"Process All {len(uploaded_files)} Invoices"):
        # Initialize client with the user's provided key
        try:
            client = genai.Client(api_key=api_key)
            extracted_data_list = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_files = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing ({i+1}/{total_files}): {uploaded_file.name}...")
                
                # Dynamic MIME type detection
                if uploaded_file.name.lower().endswith(".pdf"):
                    mime_type = "application/pdf"
                else:
                    mime_type = "image/png"
                    
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[
                            types.Part.from_bytes(
                                data=uploaded_file.read(),
                                mime_type=mime_type,
                            ),
                            "Extract the structured invoice details from this document."
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=InvoiceSchema,
                        ),
                    )
                    
                    parsed_invoice = InvoiceSchema.model_validate_json(response.text)
                    data_dict = parsed_invoice.model_dump()
                    data_dict["source_file"] = uploaded_file.name
                    extracted_data_list.append(data_dict)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / total_files)
                
            status_text.text("Processing complete!")
            
            # Display Results and Download Button
            if extracted_data_list:
                st.success(f"Successfully extracted {len(extracted_data_list)} invoices!")
                
                df = pd.DataFrame(extracted_data_list)
                st.dataframe(df)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Complete CSV Spreadsheet",
                    data=csv,
                    file_name="processed_batch_invoices.csv",
                    mime="text/csv",
                )
                
        except Exception as e:
            st.error(f"Invalid API key or connection error: {e}")
            
elif not api_key:
    st.warning("👈 Please enter your Gemini API key in the sidebar to unlock the processor.")