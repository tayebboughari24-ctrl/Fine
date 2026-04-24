import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import torch
from datetime import datetime
import io
import time

# --- 1. إعدادات الصفحة ---
st.set_page_config(
    page_title="Sentify Pro | AI Sentiment Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. تصحيح كود التنسيق (CSS) ---
# تم تصحيح unsafe_allow_html هنا
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
    .sentiment-card {
        padding: 25px;
        border-radius: 18px;
        border-left: 8px solid;
        margin-bottom: 25px;
        background-color: white;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    }
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        height: 3em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
</style>
""", unsafe_allow_html=True) 

# --- 3. محرك الذكاء الاصطناعي ---
@st.cache_resource(show_spinner="جاري تحميل محرك الذكاء الاصطناعي...")
def load_sentiment_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
        truncation=True
    )

class SentimentAnalyzer:
    def __init__(self):
        self.engine = load_sentiment_pipeline()

    def analyze(self, texts):
        return self.engine(texts, batch_size=16)

# --- 4. إدارة الجلسة والدوال المساعدة ---
if 'history' not in st.session_state:
    st.session_state.history = []

def save_to_history(text, label, score):
    st.session_state.history.append({
        "الوقت": datetime.now().strftime("%H:%M:%S"),
        "النص": text[:50] + "...",
        "التصنيف": label,
        "الثقة": f"{score:.2%}"
    })

# --- 5. واجهة المستخدم ---
def main():
    analyzer = SentimentAnalyzer()
    
    with st.sidebar:
        st.title("Sentify Pro")
        st.image("https://cdn-icons-png.flaticon.com/512/8653/8653200.png", width=100)
        mode = st.selectbox("🎯 اختر الوضع", ["التحليل الفوري", "معالجة ملفات", "السجل"])
        if st.button("مسح البيانات 🗑️", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    if mode == "التحليل الفوري":
        st.title("🚀 AI Sentiment Analysis")
        col_in, col_out = st.columns([1.2, 0.8], gap="large")

        with col_in:
            user_input = st.text_area("أدخل النص (الإنجليزية):", height=200)
            if st.button("تحليل الآن 🔍", type="primary", use_container_width=True):
                if user_input.strip():
                    with st.spinner("جاري التحليل..."):
                        res = analyzer.analyze([user_input])[0]
                        st.session_state.last_res = res
                        save_to_history(user_input, res['label'], res['score'])
                else:
                    st.warning("الرجاء إدخال نص.")

        with col_out:
            if 'last_res' in st.session_state:
                res = st.session_state.last_res
                color = "#10b981" if res['label'] == "POSITIVE" else "#ef4444"
                st.markdown(f"""
                <div class="sentiment-card" style="border-color: {color};">
                    <h2 style="color: {color};">{res['label']}</h2>
                    <p>Confidence: {res['score']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = res['score'] * 100,
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': color}}
                ))
                fig.update_layout(height=250, margin=dict(t=0, b=0, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

    elif mode == "معالجة ملفات":
        st.title("📂 Batch Processing")
        uploaded_file = st.file_uploader("ارفع ملف CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            col = st.selectbox("اختر عمود النص:", df.columns)
            if st.button("بدء المعالجة 🚀"):
                results = analyzer.analyze(df[col].astype(str).tolist())
                df['Sentiment'] = [r['label'] for r in results]
                df['Score'] = [r['score'] for r in results]
                st.success("اكتمل التحليل!")
                st.dataframe(df.head())
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("تحميل النتائج", csv, "results.csv", "text/csv")

    elif mode == "السجل":
        st.title("📜 History")
        if st.session_state.history:
            st.table(pd.DataFrame(st.session_state.history))
        else:
            st.info("السجل فارغ.")

if __name__ == "__main__":
    main()
        
