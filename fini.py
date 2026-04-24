import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
from PIL import Image
import torch
from datetime import datetime
import io
import time

# --- 1. SETTINGS & STYLING ---
st.set_page_config(
    page_title="Sentify Multi-Modal AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .sentiment-card {
        padding: 25px;
        border-radius: 18px;
        border-left: 8px solid;
        margin-bottom: 25px;
        background-color: white;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        height: 3.2em;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
</style>
""", unsafe_allow_html=True)

# --- 2. AI ENGINES (CACHED) ---

@st.cache_resource(show_spinner="Loading Text Intelligence...")
def load_text_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

@st.cache_resource(show_spinner="Loading Vision Intelligence...")
def load_image_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-classification", model="google/vit-base-patch16-224", device=device)

# --- 3. SESSION MANAGEMENT ---
if 'history' not in st.session_state:
    st.session_state.history = []

def add_to_history(input_type, content, result):
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Type": input_type,
        "Preview": content[:40] + "...",
        "Outcome": result
    })

# --- 4. MAIN APP STRUCTURE ---
def main():
    # Load models
    text_model = load_text_model()
    image_model = load_image_model()

    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>Sentify AI</h1>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/8653/8653200.png", width=100)
        st.divider()
        
        mode = st.selectbox("🎯 Select Analysis Mode", 
                            ["Text Analysis", "Batch File Processing", "Image Analysis", "History Logs"])
        
        st.divider()
        if st.button("Clear All Data 🗑️"):
            st.session_state.history = []
            st.rerun()

    # --- MODE 1: TEXT ANALYSIS ---
    if mode == "Text Analysis":
        st.title("📝 Text Sentiment Analysis")
        col_in, col_out = st.columns([1.2, 0.8], gap="large")

        with col_in:
            user_text = st.text_area("Enter English text:", height=200, placeholder="Type something positive or negative...")
            if st.button("Analyze Text 🔍", type="primary"):
                if user_text.strip():
                    with st.spinner("Analyzing..."):
                        res = text_model(user_text)[0]
                        st.session_state.last_text_res = res
                        add_to_history("Text", user_text, res['label'])
                else:
                    st.warning("Please enter text.")

        with col_out:
            if 'last_text_res' in st.session_state:
                res = st.session_state.last_text_res
                color = "#10b981" if res['label'] == "POSITIVE" else "#ef4444"
                st.markdown(f"""<div class="sentiment-card" style="border-color: {color};">
                    <h2 style="color: {color};">{res['label']}</h2>
                    <p>Confidence: {res['score']:.2%}</p></div>""", unsafe_allow_html=True)
                fig = go.Figure(go.Indicator(mode="gauge+number", value=res['score']*100, 
                                            gauge={'axis':{'range':[0,100]}, 'bar':{'color':color}}))
                st.plotly_chart(fig, use_container_width=True)

    # --- MODE 2: BATCH FILE PROCESSING ---
    elif mode == "Batch File Processing":
        st.title("📂 Batch Data Processing")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            col = st.selectbox("Select Text Column:", df.columns)
            if st.button("Process Batch 🚀"):
                with st.status("Processing rows...") as s:
                    results = text_model(df[col].astype(str).tolist(), batch_size=16)
                    df['Sentiment'] = [r['label'] for r in results]
                    df['Score'] = [r['score'] for r in results]
                    s.update(label="Complete!", state="complete")
                st.dataframe(df.head())
                st.download_button("Download Results 📥", df.to_csv(index=False), "results.csv")

    # --- MODE 3: IMAGE ANALYSIS (NEW FEATURE) ---
    elif mode == "Image Analysis":
        st.title("🖼️ Image AI Vision")
        st.markdown("Identify objects and scenes using Vision Transformer.")
        
        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            if img_file:
                st.image(img_file, use_container_width=True, caption="Uploaded Image")

        with col_res:
            if img_file:
                if st.button("Analyze Image 🔍", type="primary"):
                    with st.spinner("Visualizing..."):
                        img = Image.open(img_file)
                        predictions = image_model(img)
                        top = predictions[0]
                        
                        color = "#10b981" if top['score'] > 0.5 else "#f59e0b"
                        st.markdown(f"""<div class="sentiment-card" style="border-color: {color};">
                            <h3 style="color: {color};">Top Prediction:</h3>
                            <h1 style="text-transform: capitalize;">{top['label']}</h1>
                            <p>Confidence: {top['score']:.2%}</p></div>""", unsafe_allow_html=True)
                        
                        df_v = pd.DataFrame(predictions)
                        st.plotly_chart(px.bar(df_v, x='score', y='label', orientation='h', color='score'), use_container_width=True)
                        add_to_history("Image", img_file.name, top['label'])
            else:
                st.info("Upload an image to start.")

    # --- MODE 4: HISTORY ---
    elif mode == "History Logs":
        st.title("📜 Operation History")
        if st.session_state.history:
            st.table(pd.DataFrame(st.session_state.history))
        else:
            st.info("No records found.")

if __name__ == "__main__":
    main()
        
