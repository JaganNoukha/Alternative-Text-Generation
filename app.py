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
# IMPORTANT: For deployment, use Streamlit Secrets or environment variables
# For local testing, you can create a .streamlit/secrets.toml file with gemini_api_key = "YOUR_KEY"

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
        return "ERROR: Gemini API Key not provided. Please enter it in the sidebar."

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
    page_icon="",
    layout="wide"
)

st.title("PDF Image Analyzer with Google Gemini")
st.markdown("Upload a PDF to extract images and get detailed descriptions using the Gemini LLM.")

# Sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    
    # Retrieve API key from environment variables or Streamlit secrets first
    default_api_key_from_env_or_secrets = os.getenv("GEMINI_API_KEY", st.secrets.get("gemini_api_key", "AIzaSyDhgopVip3JtaG3ytNJGRlqi1dmxn1xbvQ"))
    
    # Text input for API key, pre-filled if available from environment/secrets
    api_key_input = st.text_input(
        "Google Gemini API Key",
        type="password",
        #value=default_api_key_from_env_or_secrets, 
        placeholder="Enter your API key here", 
        help="Enter your Google Gemini API key. If left blank, the key from environment variables or Streamlit secrets will be used (if set)."
    )

    current_gemini_api_key = api_key_input

# --- NEW: Create a placeholder for the download button at the top ---
download_placeholder = st.empty() # This creates an empty slot where content can be placed later

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# st.caption("**:red[Note:]** The maximum upload size for a PDF file is 1 GB. Ensure `maxUploadSize = 1024` is set in `.streamlit/config.toml`.")


if uploaded_file is not None:
    if not current_gemini_api_key:
        st.warning("Please enter your Google Gemini API key in the sidebar to proceed with image analysis.")
    else:
        st.write(f"Processing PDF: **{uploaded_file.name}**")

        # Read PDF bytes
        pdf_bytes = uploaded_file.read()

        analysis_results_for_excel = []

        with st.spinner("Extracting images from PDF... This may take a moment."):
            extracted_data = extract_images_for_gemini_analysis(pdf_bytes)

        if extracted_data:
            st.subheader("Image Analysis Results:")
            
            progress_text = "Analyzing images with Gemini LLM. Please wait..."
            my_bar = st.progress(0, text=progress_text)
            
            for i, item in enumerate(extracted_data):
                page_number = item['page_number']
                image_index = item['image_index']
                image_bytes = item['image_bytes']
                mime_type = item['mime_type']
                identifier = f"Page {page_number} Image {image_index}"

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(image_bytes, caption=f"Original Image ({identifier})", use_column_width=True)
                
                with col2:
                    my_bar.progress((i + 1) / len(extracted_data), text=f"Analyzing {identifier}...")
                    gemini_response = get_image_description_from_gemini(
                        image_bytes,
                        mime_type,
                        current_gemini_api_key,
                        identifier=identifier
                    )
                    st.write(f"**Description for {identifier}:**")
                    st.markdown(gemini_response)

                analysis_results_for_excel.append({
                    'Image Identifier': identifier,
                    'Gemini Description': gemini_response
                })
                
                time.sleep(1)

            my_bar.empty()
            st.success("Image analysis complete!")

            # --- Move the Download Results section here, rendering into the placeholder ---
            if analysis_results_for_excel:
                try:
                    df = pd.DataFrame(analysis_results_for_excel)
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False)
                    excel_buffer.seek(0)

                    # --- Render into the placeholder created earlier ---
                    with download_placeholder.container(): # Use a container within the empty slot
                        # To align right, you might need to use columns here as well
                        # Let's create a row with two columns, one empty for spacing
                        _, download_col = st.columns([3, 1]) # Adjust ratio as needed for alignment
                        with download_col:
                            st.download_button(
                                label="Download Analysis (Excel)", # Shorter label for top-right
                                data=excel_buffer,
                                file_name="image_analysis_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    # st.info("Analysis results are ready for download.") # You can keep or remove this info message

                except Exception as e:
                    st.error(f"Error preparing Excel file for download: {e}")

        else:
            st.warning("No images were extracted from the PDF. Please try a different PDF.")

elif uploaded_file is None:
    st.info("Upload a PDF file to begin the analysis.")
