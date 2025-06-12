import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import base64
import requests
import time
import pandas as pd

# --- Helper Function to Get API Key ---
def get_gemini_api_key():
    try:
        secret = st.secrets.get("gemini_api_key")
    except Exception:
        secret = None
    return st.session_state.api_key_input_value or os.getenv("GEMINI_API_KEY") or secret

# --- Session State Initialization ---
if 'uploaded_pdf_name' not in st.session_state:
    st.session_state.uploaded_pdf_name = None
if 'extracted_images' not in st.session_state:
    st.session_state.extracted_images = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'api_key_input_value' not in st.session_state:
    st.session_state.api_key_input_value = ""

# --- Image Extraction Function ---
@st.cache_data
def extract_images_for_gemini_analysis(pdf_bytes):
    extracted_data = []
    try:
        pdf_file = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        st.error(f"Error opening PDF: {e}")
        return []

    st.info("Starting in-memory image extraction from PDF...")

    for page_number in range(len(pdf_file)):
        page = pdf_file[page_number]
        images = page.get_images(full=True)

        if not images:
            st.warning(f"No images found on page {page_number + 1}.")
            continue

        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = (base_image.get("ext") or "").lower()

                if image_ext not in ['png', 'jpeg', 'jpg', 'gif', 'bmp', 'webp']:
                    st.warning(f"Unsupported image format '{image_ext}' on page {page_number+1}. Defaulting to 'image/png'.")
                    mime_type = "image/png"
                else:
                    mime_type = f"image/{image_ext}"
                    if mime_type == "image/jpg":
                        mime_type = "image/jpeg"

                extracted_data.append({
                    'page_number': page_number + 1,
                    'image_index': img_index + 1,
                    'image_bytes': image_bytes,
                    'mime_type': mime_type
                })
            except Exception as e:
                st.error(f"Error processing image {img_index + 1} on page {page_number + 1}: {e}")

    pdf_file.close()
    st.success(f"In-memory image extraction complete. Found {len(extracted_data)} images.")
    return extracted_data

# --- Gemini API Call ---
def get_image_description_from_gemini(image_bytes, mime_type, api_key, identifier=""):
    if not api_key:
        return "ERROR: Gemini API Key not provided."

    try:
        image_data_base64 = base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        return f"Error encoding image bytes for {identifier}: {e}"

    prompt_text = (
        "Analyze this image in detail. Describe what is in the image, "
        "what the image says about, and its overall significance. "
        "Pay close attention to any text visible within the image itself."
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": image_data_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 400
        }
    }

    headers = {
        "Content-Type": "application/json"
    }

    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    try:
        response = requests.post(f"{GEMINI_API_URL}?key={api_key}", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        if (result.get("candidates") and 
            result["candidates"][0].get("content") and 
            result["candidates"][0]["content"].get("parts")):
            return result["candidates"][0]["content"]["parts"][0].get("text", "No text response found.")
        else:
            reason = result.get("promptFeedback", {}).get("blockReason", "Unknown")
            return f"No valid response for {identifier}. Blocked due to: {reason}."
    except requests.exceptions.RequestException as e:
        return f"Connection error for {identifier}: {e}"
    except Exception as e:
        return f"Unexpected error for {identifier}: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Image Analyzer with Gemini", page_icon="📄", layout="wide")

st.title("PDF Image Analyzer with Google Gemini")
st.markdown("Upload a PDF to extract images and get detailed descriptions using the Gemini LLM.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Google Gemini API Key", type="password",
                                  value=st.session_state.api_key_input_value,
                                  placeholder="Enter API key",
                                  help="Optional if set in environment or Streamlit secrets.")
    st.session_state.api_key_input_value = api_key_input

    # Debug Info
    st.markdown("### Debug Info")
    st.write("API Key Source:", 
        "Manual" if api_key_input else 
        "Environment" if os.getenv("GEMINI_API_KEY") else 
        "Secrets" if get_gemini_api_key() else "None")

# --- Download Placeholder ---
download_placeholder = st.empty()

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and (uploaded_file.name != st.session_state.uploaded_pdf_name or not st.session_state.extracted_images):
    st.session_state.uploaded_pdf_name = uploaded_file.name
    st.session_state.extracted_images = []
    st.session_state.analysis_results = []

    current_api_key = get_gemini_api_key()

    if not current_api_key:
        st.warning("Please provide your Gemini API key in the sidebar.")
    else:
        st.write(f"Processing PDF: **{uploaded_file.name}**")
        pdf_bytes = uploaded_file.read()

        with st.spinner("Extracting images..."):
            st.session_state.extracted_images = extract_images_for_gemini_analysis(pdf_bytes)

        if st.session_state.extracted_images:
            st.subheader("Image Analysis Results")
            my_bar = st.progress(0, text="Analyzing images...")

            results = []
            for i, item in enumerate(st.session_state.extracted_images):
                identifier = f"Page {item['page_number']} Image {item['image_index']}"
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(item['image_bytes'], caption=identifier, use_column_width=True)
                with col2:
                    my_bar.progress((i + 1) / len(st.session_state.extracted_images), text=f"Analyzing {identifier}...")
                    description = get_image_description_from_gemini(item['image_bytes'], item['mime_type'], current_api_key, identifier)
                    st.write(f"**{identifier}:**")
                    st.markdown(description)
                results.append({'Image Identifier': identifier, 'Gemini Description': description})
                time.sleep(0.5)

            st.session_state.analysis_results = results
            my_bar.empty()
            st.success("Image analysis complete!")

        else:
            st.warning("No images were extracted from the PDF. Try another file.")

elif uploaded_file is None and st.session_state.uploaded_pdf_name is None:
    st.info("Upload a PDF file to begin.")

# --- Download Button & Results Rendering ---
if st.session_state.analysis_results:
    if uploaded_file and uploaded_file.name == st.session_state.uploaded_pdf_name:
        st.subheader("Image Analysis Results")
        for item in st.session_state.analysis_results:
            image_info = next((img for img in st.session_state.extracted_images 
                               if f"Page {img['page_number']} Image {img['image_index']}" == item['Image Identifier']), None)
            if image_info:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image_info['image_bytes'], caption=item['Image Identifier'], use_column_width=True)
                with col2:
                    st.write(f"**{item['Image Identifier']}:**")
                    st.markdown(item['Gemini Description'])

    try:
        df = pd.DataFrame(st.session_state.analysis_results)
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        with download_placeholder.container():
            _, download_col = st.columns([3, 1])
            with download_col:
                st.download_button(
                    label="Download Analysis (Excel)",
                    data=excel_buffer,
                    file_name="image_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel_button"
                )
    except Exception as e:
        st.error(f"Error preparing Excel file for download: {e}")
