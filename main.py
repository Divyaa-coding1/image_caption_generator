import io
import os 
import cv2 
import numpy as np 

from PIL import Image
import streamlit as st 

from caption_generation import MultiModalCaptionGenerator
from caption_history import CaptionHistory
from caption_overlay import ImageCaptionOverlay
from dotenv import load_dotenv
import glob

load_dotenv()

# Function to get available fonts from the fonts folder
def get_available_fonts():
    """Get list of available TrueType fonts from the fonts folder"""
    fonts_dict = {}

    # Check if fonts folder exists
    if os.path.exists("fonts"):
        # Find all .ttf files in the fonts folder
        font_files = glob.glob("fonts/*.ttf") + glob.glob("fonts/*.TTF")

        for font_path in font_files:
            # Extract font name from filename (remove path and extension)
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            # Make it more readable (replace hyphens/underscores with spaces)
            font_name = font_name.replace("-", " ").replace("_", " ")
            fonts_dict[font_name] = font_path

    # Sort fonts alphabetically and add Default first
    sorted_fonts = dict(sorted(fonts_dict.items()))
    final_fonts = {"Default (Streamlit)": None}
    final_fonts.update(sorted_fonts)

    return final_fonts

openai_key = os.getenv("OPENAI_API_ICG")
groq_key = os.getenv("GROQ_API_ICG")
gemini_key = os.getenv("GEMINI_API_ICG")


st.set_page_config(
    page_title="Multi-Model Image Caption Generator",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Multi-Model Image Caption Generator")
st.markdown("Generate captions using OpenAI GPT-5 Nano, Google Gemini, and GROQ models.")

# Initialize session state 
if "caption_history" not in st.session_state:
    st.session_state.caption_history = CaptionHistory()

if "caption_generator" not in st.session_state:
    st.session_state.caption_generator = MultiModalCaptionGenerator()

# Sidebar for API configuration 
with st.sidebar:
    st.header("🍋‍🟩 API Configureation")

    # Show API status
    if openai_key:
        st.success("✅ OPENAI API KEY accessed from environment.")
    else:
        st.warning("⚠️ OPENAI API KEY not found in environment.")
    
    if groq_key:
        st.success("✅ GROQ API KEY accessed from environment.")
    else:
        st.warning("⚠️ GROQ API KEY not found in environment.")
    
    if gemini_key:
        st.success("✅ GEMINI API KEY accessed from environment.")
    else:
        st.warning("⚠️ GEMINI API KEY not found in environment.")

    if st.button("Configure APIs"):
        try:
            st.session_state.caption_generator.configure_apis(
                openai_key=openai_key,
                groq_key=groq_key,
                gemini_key=gemini_key
            )
            st.success("✅ APIs configured successfully.")
        except Exception as e:
            st.error(f"❌ Error configuring APIs: {e}")

    st.markdown("---")

    # Caption overlay settings
    st.header("🧑‍🎨 Caption Settings")
    caption_method = st.selectbox(
        "Caption Method",
        ["Overlay on Image", "Background behind Image"]
    )

    # Get available fonts
    available_fonts = get_available_fonts()

    # Font selection (common for both methods)
    selected_font_name = st.selectbox(
        "Font Style",
        options=list(available_fonts.keys()),
        index=0,
        help="Select a font from your system fonts or use the default"
    )
    selected_font_path = available_fonts[selected_font_name]

    if caption_method == "Overlay on Image":
        position = st.selectbox("Position", ["Bottom", "Center", "Top"])
        font_size = st.slider("Font Size", 0.5, 3.0, value=1.0, step=0.1)
        thickness = st.slider("Thickness", 1, 5, value=2, step=1)
    else:
        bg_color = st.color_picker("Background Color", value="#000000")
        text_color = st.color_picker("Text Color", value="#FFFFFF")
        margin = st.slider("Margin", 20, 100, value=50, step=10)
        pil_font_size = st.slider("Font Size", 12, 72, value=24, step=2)

    st.markdown("---")

    # History Management 
    st.header("📃 Caption History")
    if st.button("Clear History"):
        st.session_state.caption_history.clear_history()
        st.success("✅ History cleared successfully.")

# Main content Area 
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📷 Upload Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])

    if uploaded_file is not None:
        # Display original image 
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", width="content")

        # Model selection 
        st.header("🤖 Select Model")
        models = {
            "OpenAI GPT-5 Nano": "openai",
            "Google  GEMINI 2.5 Flash Lite": "gemini",
            "GROQ VISION": "groq"
        }

        selected_model = st.selectbox("Choose a model", list(models.keys()))

        if st.button("Generate Caption", type="primary"):
            try:
                model_key = models[selected_model]
                caption = ""

                with st.spinner("Generating caption with {selected_model}..."):
                    if model_key == "openai":
                        caption = st.session_state.caption_generator.generate_caption_openai(image)
                    elif model_key == "gemini":
                        caption = st.session_state.caption_generator.generate_caption_gemini(image)
                    elif model_key == "groq":
                        caption = st.session_state.caption_generator.generate_caption_groq(image)

                    st.session_state.current_caption = caption
                    st.session_state.current_image = image
                    st.session_state.current_model = selected_model

                    # Add to history
                    st.session_state.caption_history.add_interaction(
                        uploaded_file.name,
                        selected_model,
                        caption
                    )
            except Exception as e:
                st.error(f"Error generating caption: {str(e)}")

with col2:
    st.header("🪄✨ Generated Captions and Preview")

    if hasattr(st.session_state, "current_caption"):
        st.text_area("Generated Caption", st.session_state.current_caption, height=100)
        
        # Generate Preview with Caption 
        if hasattr(st.session_state, "current_image"):
            # Convert PIL to OpenCV format 
            cv_image = cv2.cvtColor(np.array(st.session_state.current_image), cv2.COLOR_RGB2BGR)

            if caption_method == "Overlay on Image":
                result_image = ImageCaptionOverlay.add_caption_overlay(
                    cv_image,
                    st.session_state.current_caption,
                    position=position,
                    font_size=font_size,
                    thickness=thickness,
                    font_path=selected_font_path
                )
            else:
                # Convert hex colors to RGB colors
                bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
                text_rgb = tuple(int(text_color[i:i+2], 16) for i in (1, 3, 5))

                result_image = ImageCaptionOverlay.add_caption_background(
                    cv_image,
                    st.session_state.current_caption,
                    font_path=selected_font_path,
                    font_size=pil_font_size,
                    background_color=bg_rgb,
                    text_color=text_rgb,
                    margin=margin
                )

            # Convert back to PIL for display
            result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            st.image(result_pil, caption="Image with Caption", use_container_width=True)

            # Download Button 
            img_buffer = io.BytesIO()
            result_pil.save(img_buffer, format="PNG")

            st.download_button(
                label="🔽 Download Image with Caption",
                data=img_buffer.getvalue(),
                file_name=f"captioned_{uploaded_file.name if uploaded_file else 'image'}",
                mime="image/png"
            
            )

# History display 
if getattr(st.session_state, "show_history", False):
    st.markdown("---")
    st.header("📟️ Caption Generation History")

    history = st.session_state.caption_history.get_history()

    if history:
        for index, item in enumerate(reversed(history[-10:])):
            with st.expander(f"{item['timestamp'][:19]} - {item['image_name']} ({item['model']})"):
                st.write(f"**Model:** {item['model']}") 
                st.write(f"**Image: {item['image_name']}**")
                st.write(f"**Caption: {item['caption']}**")
                st.write(f"**Timestamp: {item['timestamp']}**")
    else:
        st.info("No caption history available.")

# Footer 
st.markdown("---")
st.markdown("""
<div style="text-align: center">
<p>Built with Streamlit, LangChain, OpenCV, and multi-model AI APIs</p>
<p>Supports OpenAI GPT-5 Nano, GROQ VISION, and Google Gemini 2.5 Flash Lite</p>
</div>
""", unsafe_allow_html=True)