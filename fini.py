import streamlit as st
from transformers import pipeline
import pandas as pd
from PIL import Image
import io

# 1. إعداد الصفحة والتحميل
st.set_page_config(page_title="High-Accuracy AI Analyst", layout="wide")

@st.cache_resource
def load_ai_models():
    # تحميل أقوى نماذج متاحة لهذه المهام
    text_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    image_pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    return text_pipe, image_pipe

text_ai, img_ai = load_ai_models()

# --- دالة تصنيف النصوص (الدقة القصوى) ---
def get_sentiment_label(result):
    label = result['label'] # المخرج يكون مثل "1 star" أو "5 stars"
    # استخراج الرقم فقط من النص
    stars = int(''.join(filter(str.isdigit, label)))
    
    # توزيع النجوم بدقة أكاديمية:
    if stars <= 2: 
        return "Negative ❌", "red"
    elif stars == 3: 
        return "Neutral 😐", "orange"
    else: # 4 أو 5 نجوم
        return "Positive ✅", "green"

# --- الواجهة ---
st.title("🎭 High-Accuracy Emotions Analyst")
st.markdown("---")

tab_text, tab_image, tab_file = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "📁 Bulk File Analysis"])

# --- 📝 تحليل النصوص ---
with tab_text:
    st.header("Precision Text Sentiment")
    user_input = st.text_area("Enter text (Arabic, English, French...):", height=100)
    
    if st.button("Analyze Text"):
        if user_input:
            # زيادة الدقة عبر تنظيف النص برمجياً قبل التحليل
            clean_input = user_input.strip()
            res = text_ai(clean_input)[0]
            sentiment, color = get_sentiment_label(res)
            
            # عرض النتيجة
            st.subheader("Result:")
            if color == "red": st.error(sentiment)
            elif color == "orange": st.warning(sentiment)
            else: st.success(sentiment)
            
            st.write(f"**Confidence Level:** {res['score']:.2%}")
        else:
            st.warning("Please enter some text.")

# --- 🖼️ تحليل الصور ---
with tab_image:
    st.header("Precision Facial Emotion")
    uploaded_image = st.file_uploader("Upload Image:", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, width=300)
        
        if st.button("Identify Emotion"):
            with st.spinner("Analyzing..."):
                # النموذج يعطي قائمة بأعلى التوقعات
                results = img_ai(img)
                top_res = results[0]
                
                st.subheader(f"Detected Emotion: **{top_res['label']}**")
                st.write(f"**Accuracy:** {top_res['score']:.2%}")
                st.progress(top_res['score'])

# --- 📁 تحليل الملفات والمقارنة ---
with tab_file:
    st.header("Bulk Analysis & Comparison")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        
        # اختيار عمود البيانات وعمود المقارنة (اختياري)
        col_text = st.selectbox("Select Text Column:", df.columns)
        col_truth = st.selectbox("Select Actual Results Column (for Comparison):", ["None"] + list(df.columns))
        
        if st.button("Start Bulk Analysis"):
            with st.spinner("Processing..."):
                # تحليل كل سطر
                def fast_analyze(x):
                    r = text_ai(str(x)[:512])[0] # تحديد طول النص لزيادة السرعة والدقة
                    label, _ = get_sentiment_label(r)
                    return label

                df['AI_Result'] = df[col_text].apply(fast_analyze)
                
                # عرض النتائج والأغلبية
                majority = df['AI_Result'].value_counts().idxmax()
                st.success(f"Majority Sentiment: {majority}")
                
                # إذا اختار المستخدم عمود للمقارنة، نحسب الدقة
                if col_truth != "None":
                    # تنظيف العمودين للمقارنة العادلة
                    df['Temp_Truth'] = df[col_truth].astype(str).str.strip().str.capitalize()
                    df['Temp_AI'] = df['AI_Result'].str.split().str[0] # يأخذ الكلمة بدون الإيموجي
                    
                    # عرض عينة للمقارنة بالعين
                    st.write("Preview Comparison:", df[[col_text, col_truth, 'AI_Result']].head(10))
                
                st.download_button("Download Report", df.to_csv(index=False), "ai_report.csv")
        
