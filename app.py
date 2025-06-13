import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import base64
import time
import pandas as pd
import openai
from dotenv import load_dotenv
load_dotenv()

# --- API Key Handling ---
def get_openai_api_key():
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    return st.session_state.api_key_input_value

# --- Session State Initialization ---
for key in [
    'uploaded_pdf_name', 'extracted_images', 'analysis_results',
    'api_key_input_value', 'custom_prompt']:
    if key not in st.session_state:
        st.session_state[key] = [] if 'results' in key or 'images' in key else ""

# --- Default Prompt ---
DEFAULT_PROMPT = (
    """
    Analyze the image thoroughly and describe it in a story format. Include details about objects, subjects, actions, and any visible text. Interpret what the image communicates and its significance. Avoid any mention of colors.
    Also, provide a sharp, concise summary under 300 characters that captures the core message of the image.
    """
)

# --- Extract Images from PDF ---
@st.cache_data
def extract_images_for_analysis(pdf_bytes):
    extracted_data = []
    try:
        pdf_file = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        st.error(f"Error opening PDF: {e}")
        return []

    for page_number in range(len(pdf_file)):
        page = pdf_file[page_number]
        images = page.get_images(full=True)
        if not images:
            continue

        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = (base_image.get("ext") or "").lower()

                mime_type = f"image/{image_ext}" if image_ext else "image/png"
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
    return extracted_data

# --- OpenAI Vision Analysis ---
def get_image_description_from_openai(image_bytes, prompt_text, api_key, identifier=""):
    if not api_key:
        return "ERROR: OpenAI API Key not provided."
    if not prompt_text:
        return "ERROR: Analysis prompt cannot be empty."

    openai.api_key = api_key

    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"

        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            temperature=0.4,
            max_tokens=400
        )

        full_content = response.choices[0].message.content.strip()

        # Extract summary if available
        summary_marker = "Summary:"
        if summary_marker in full_content:
            parts = full_content.split(summary_marker, 1)
            long_text = parts[0].strip()
            summary_text = f"\n{parts[1].strip()}"
        else:
            long_text = full_content
            summary_text = (full_content[:300] + "...") if len(full_content) > 300 else full_content

        return long_text, summary_text

    except Exception as e:
        return f"OpenAI error for {identifier}: {e}", ""

# --- Streamlit UI Setup ---
st.set_page_config(page_title="PDF Image Analyzer with OpenAI GPT-4o", page_icon="📄", layout="wide")
st.title("📄 PDF Image Analyzer with OpenAI GPT-4o")
st.markdown("Upload a PDF to extract images and get detailed descriptions using OpenAI's vision model.")

excel_download_placeholder = st.empty()

# --- Sidebar ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key_input = st.text_input("OpenAI API Key", type="password",
                                  value=st.session_state.api_key_input_value,
                                  placeholder="Enter OpenAI API key")
    st.session_state.api_key_input_value = api_key_input

    st.markdown("### API Key Source")
    current_api_key_check = get_openai_api_key()
    if current_api_key_check and current_api_key_check == os.getenv("OPENAI_API_KEY"):
        st.info("API Key Source: Environment Variable")
    elif api_key_input:
        st.info("API Key Source: Manual Input")
    else:
        st.warning("API Key Source: None")

# --- Prompt Input ---
st.subheader("📝 Analysis Prompt")
st.session_state.custom_prompt = st.text_area(
    "Enter your prompt for image analysis:",
    value=st.session_state.custom_prompt or DEFAULT_PROMPT,
    height=150
)

# --- File Upload ---
uploaded_file = st.file_uploader("📎 Upload a PDF", type="pdf")

if uploaded_file and (uploaded_file.name != st.session_state.uploaded_pdf_name or not st.session_state.extracted_images):
    st.session_state.uploaded_pdf_name = uploaded_file.name
    st.session_state.extracted_images = []
    st.session_state.analysis_results = []

    current_api_key = get_openai_api_key()
    current_analysis_prompt = st.session_state.custom_prompt

    if current_api_key and current_analysis_prompt:
        pdf_bytes = uploaded_file.read()

        with st.spinner("Extracting images from PDF..."):
            st.session_state.extracted_images = extract_images_for_analysis(pdf_bytes)

        if st.session_state.extracted_images:
            st.subheader("🔍 Image Analysis Results")
            my_bar = st.progress(0, text="Analyzing images...")

            for i, item in enumerate(st.session_state.extracted_images):
                identifier = f"Page {item['page_number']} Image {item['image_index']}"

                description_long, description_short = get_image_description_from_openai(
                    item['image_bytes'], current_analysis_prompt, current_api_key, identifier
                )

                st.session_state.analysis_results.append({
                    'Page Number': item['page_number'],
                    'Alt Short Text': description_short,
                    'Alt Long Text': description_long
                })

                my_bar.progress((i + 1) / len(st.session_state.extracted_images), text=f"Analyzing {identifier}...")

                left_col, right_col = st.columns([1, 2])
                with left_col:
                    st.image(item['image_bytes'], caption=identifier, use_column_width=True)
                with right_col:
                    st.markdown(f"**Short Alt Text:** {description_short}")
                    st.markdown(f"**Long Alt Text:** {description_long}")

                time.sleep(0.4)

            my_bar.empty()
            st.success("✅ Image analysis complete!")

# --- Excel Download ---
if st.session_state.analysis_results:
    df = pd.DataFrame(st.session_state.analysis_results)
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)

    excel_download_placeholder.markdown("""
        <div style='text-align: right;'>
            <a download='image_analysis_results.xlsx' href='data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{}' target='_blank'>
                <button style='font-size:16px;padding:10px 20px;'>📥 Download Analysis (Excel)</button>
            </a>
        </div>
    """.format(base64.b64encode(excel_buffer.read()).decode()), unsafe_allow_html=True)
