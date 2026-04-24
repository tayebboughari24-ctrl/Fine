import streamlit as st
from transformers import pipeline
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score # لإضافة خاصية حساب الدقة

# 1. Page Setup
st.set_page_config(page_title="Emotions Analyst", layout="wide")

@st.cache_resource
def load_ai_models():
    text_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    image_pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    return text_pipe, image_pipe

text_ai, img_ai = load_ai_models()

def get_sentiment_label(result):
    label = result['label']
    stars = int(label.split()[0])
    if stars <= 2: return "Negative" # أزلت الإيموجي هنا لتسهيل المقارنة مع ملفاتك
    if stars == 3: return "Neutral"
    return "Positive"

# Header
st.title("🎭 Emotions Analyst & Benchmarker")
st.markdown("---")

tab_text, tab_image, tab_file = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "📁 Bulk Comparison"])

# --- TAB 1 & 2 (كما هي) ---
with tab_text:
    st.header("Analyze Text Sentiment")
    user_input = st.text_area("Paste your text here:")
    if st.button("Analyze Text"):
        if user_input:
            res = text_ai(user_input)[0]
            sentiment = get_sentiment_label(res)
            st.subheader(f"Result: {sentiment}")

with tab_image:
    st.header("Analyze Image Emotions")
    uploaded_image = st.file_uploader("Upload a photo...", type=['jpg', 'png', 'jpeg'])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, width=400)
        if st.button("Identify Emotion"):
            res = img_ai(img)
            st.success(f"Detected: {res[0]['label']}")

# --- TAB 3: BULK COMPARISON (المعدل للمقارنة) ---
with tab_file:
    st.header("Compare AI Results with Ground Truth")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            text_column = st.selectbox("Select Text Column (Data):", df.columns)
        with col2:
            truth_column = st.selectbox("Select Ground Truth Column (Actual Results):", df.columns)
        
        if st.button("Run Comparison Analysis"):
            with st.spinner("AI is evaluating and comparing..."):
                # 1. تنظيف البيانات
                df = df.dropna(subset=[text_column, truth_column])
                
                # 2. تحليل النصوص بواسطة الذكاء الاصطناعي
                df['AI_Prediction'] = df[text_column].apply(lambda x: get_sentiment_label(text_ai(str(x))[0]))
                
                # 3. توحيد صيغة عمود النتائج الحقيقية (لضمان دقة المقارنة)
                # تحويل النتائج في ملفك لتشابه مخرجات النموذج (Capitalize)
                df[truth_column] = df[truth_column].astype(str).str.strip().str.capitalize()
                
                # 4. حساب الدقة
                acc = accuracy_score(df[truth_column], df['AI_Prediction'])
                
                st.divider()
                # عرض الدقة بشكل بارز
                st.metric(label="Model Accuracy vs Your Results", value=f"{acc:.2%}")
                
                # عرض جدول المقارنة
                st.subheader("Results Preview")
                st.dataframe(df[[text_column, truth_column, 'AI_Prediction']].head(10))
                
                # خيار تحميل النتائج المقارنة
                st.download_button("Download Comparison Report", df.to_csv(index=False), "comparison_results.csv")

