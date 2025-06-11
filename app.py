import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import base64
import requests # For making API calls
import time # For potential rate limiting
import pandas as pd # For creating and saving Excel files

# --- Configuration for Gemini API ---
# This app is designed to get the API key primarily from environment variables
# (e.g., set on an EC2 instance) or from Streamlit secrets (for Streamlit Cloud).
# A sidebar input is also provided for local testing or manual entry.

# Initialize session state variables if they don't exist
if 'uploaded_pdf_name' not in st.session_state:
    st.session_state.uploaded_pdf_name = None
if 'extracted_images' not in st.session_state:
    st.session_state.extracted_images = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'api_key_input_value' not in st.session_state:
    st.session_state.api_key_input_value = ""

# Your existing functions (slightly modified for Streamlit integration)
@st.cache_data # Cache extracted images to avoid re-extracting if inputs don't change
def extract_images_for_gemini_analysis(pdf_bytes):
    """
    Extracts images from PDF bytes directly into memory.
    Returns a list of dictionaries, each containing 'image_bytes', 'mime_type',
    'page_number', and 'image_index'.
    """
    extracted_data = []
    try:
        # Open PDF from bytes, not file path
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
                image_ext = base_image["ext"]

                if not image_ext or image_ext.lower() not in ['png', 'jpeg', 'jpg', 'gif', 'bmp', 'webp']:
                    st.warning(f"Unsupported image format '{image_ext}' for image {img_index+1} on page {page_number+1}. Defaulting to 'png' MIME type.")
                    mime_type = "image/png"
                    if image_ext.lower() in ['jpeg', 'jpg']:
                        mime_type = "image/jpeg"
                else:
                    mime_type = f"image/{image_ext.lower()}"
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

@st.cache_resource # Cache the Gemini API call if inputs are the same
def get_image_description_from_gemini(image_bytes, mime_type, api_key, identifier=""):
    """
    Sends image bytes to the Gemini LLM for analysis.
    """
    if not api_key:
        return "ERROR: Gemini API Key not provided. Please enter it in the sidebar or set the GEMINI_API_KEY environment variable."

    try:
        image_data_base64 = base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        return f"Error encoding image bytes for {identifier}: {e}"

    prompt_text = "Analyze this image in detail. Describe what is in the image, what the image says about, and its overall significance. Pay close attention to any text visible within the image itself. Be concise but informative."

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

        if result.get("candidates") and len(result["candidates"]) > 0 and \
           result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           len(result["candidates"][0]["content"]["parts"]) > 0:
            text_response = result["candidates"][0]["content"]["parts"][0].get("text", "No text response found.")
            return text_response
        else:
            error_message = f"No valid response from Gemini for image {identifier}."
            if result.get("promptFeedback") and result["promptFeedback"].get("blockReason"):
                error_message += f" Blocked due to: {result['promptFeedback']['blockReason']}."
            return error_message + f" Full response: {result}"

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Gemini API for image {identifier}: {e}"
    except Exception as e:
        return f"An unexpected error occurred during Gemini API call for image {identifier}: {e}"


# --- Streamlit UI ---
st.set_page_config(
    page_title="PDF Image Analyzer with Gemini",
    page_icon="📄",
    layout="wide"
)

st.title("PDF Image Analyzer with Google Gemini")
st.markdown("Upload a PDF to extract images and get detailed descriptions using the Gemini LLM.")

# Determine the API key priority:
# 1. Environment variable (GEMINI_API_KEY) - ideal for EC2
# 2. Streamlit Secrets (gemini_api_key in secrets.toml or Streamlit Cloud)
# 3. User input in the sidebar
env_api_key = os.getenv("GEMINI_API_KEY")
streamlit_secret_api_key = st.secrets.get("gemini_api_key")

# The effective default for the text input
initial_api_key_for_input = env_api_key or streamlit_secret_api_key or st.session_state.api_key_input_value

with st.sidebar:
    st.header("Configuration")
    
    # Text input for API key, pre-filled if available from environment/secrets
    api_key_input = st.text_input(
        "Google Gemini API Key",
        type="password",
        # value=initial_api_key_for_input, 
        placeholder="Enter your API key here", 
        help="Enter your Google Gemini API key. If left blank, the key from environment variables (GEMINI_API_KEY) or Streamlit secrets will be used (if set)."
    )
    # Update session state with the current input value
    st.session_state.api_key_input_value = api_key_input
    
    # The API key currently being used by the application logic
    # This ensures manual input overrides env/secrets if user types something
    current_gemini_api_key = st.session_state.api_key_input_value

# --- NEW: Create a placeholder for the download button at the top ---
download_placeholder = st.empty() # This creates an empty slot where content can be placed later

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Check if a new file has been uploaded or if no file was previously processed
if uploaded_file is not None and (uploaded_file.name != st.session_state.uploaded_pdf_name or not st.session_state.extracted_images):
    st.session_state.uploaded_pdf_name = uploaded_file.name
    # Clear previous results when a new PDF is uploaded
    st.session_state.extracted_images = []
    st.session_state.analysis_results = []
    
    if not current_gemini_api_key:
        st.warning("Please enter your Google Gemini API key in the sidebar or set the `GEMINI_API_KEY` environment variable on your server to proceed with image analysis.")
    else:
        st.write(f"Processing PDF: **{uploaded_file.name}**")

        # Read PDF bytes
        pdf_bytes = uploaded_file.read()

        with st.spinner("Extracting images from PDF... This may take a moment."):
            st.session_state.extracted_images = extract_images_for_gemini_analysis(pdf_bytes)

        if st.session_state.extracted_images:
            st.subheader("Image Analysis Results:")
            
            progress_text = "Analyzing images with Gemini LLM. Please wait..."
            my_bar = st.progress(0, text=progress_text)
            
            new_analysis_results = [] # Temporary list for current run
            for i, item in enumerate(st.session_state.extracted_images):
                page_number = item['page_number']
                image_index = item['image_index']
                image_bytes = item['image_bytes']
                mime_type = item['mime_type']
                identifier = f"Page {page_number} Image {image_index}"

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(image_bytes, caption=f"Original Image ({identifier})", use_column_width=True)
                
                with col2:
                    my_bar.progress((i + 1) / len(st.session_state.extracted_images), text=f"Analyzing {identifier}...")
                    gemini_response = get_image_description_from_gemini(
                        image_bytes,
                        mime_type,
                        current_gemini_api_key,
                        identifier=identifier
                    )
                    st.write(f"**Description for {identifier}:**")
                    st.markdown(gemini_response)

                new_analysis_results.append({
                    'Image Identifier': identifier,
                    'Gemini Description': gemini_response
                })
                
                # Introduce a small delay to prevent hitting API rate limits too quickly
                time.sleep(0.5) 

            st.session_state.analysis_results = new_analysis_results # Store results in session state
            
            my_bar.empty()
            st.success("Image analysis complete!")

        else:
            st.warning("No images were extracted from the PDF. Please try a different PDF.")

elif uploaded_file is None and st.session_state.uploaded_pdf_name is None:
    st.info("Upload a PDF file to begin the analysis.")

# --- Display results and download button if analysis results are in session state ---
if st.session_state.analysis_results:
    # If the app reruns and results are already present, display them
    # This block ensures results are displayed even after a rerun, as long as a PDF is "active"
    if uploaded_file is not None and uploaded_file.name == st.session_state.uploaded_pdf_name:
        # Only re-display if images were extracted and processed
        if st.session_state.extracted_images:
            st.subheader("Image Analysis Results:")
            for item in st.session_state.analysis_results:
                # Find the corresponding image from extracted_images to display
                # This part assumes extracted_images are also in session state and match
                image_info = next((img for img in st.session_state.extracted_images 
                                    if f"Page {img['page_number']} Image {img['image_index']}" == item['Image Identifier']), None)
                
                if image_info:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(image_info['image_bytes'], caption=f"Original Image ({item['Image Identifier']})", use_column_width=True)
                    with col2:
                        st.write(f"**Description for {item['Image Identifier']}:**")
                        st.markdown(item['Gemini Description'])
    
    try:
        df = pd.DataFrame(st.session_state.analysis_results)
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        # --- Render into the placeholder created earlier ---
        with download_placeholder.container(): # Use a container within the empty slot
            _, download_col = st.columns([3, 1]) # Adjust ratio as needed for alignment
            with download_col:
                st.download_button(
                    label="Download Analysis (Excel)", # Shorter label for top-right
                    data=excel_buffer,
                    file_name="image_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel_button" # Add a unique key
                )

    except Exception as e:
        st.error(f"Error preparing Excel file for download: {e}")
