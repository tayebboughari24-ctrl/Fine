import streamlit as st
import pandas as pd
import io
import re
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import pipeline

# ==========================================
# 1. CONFIGURATION & PAGE STYLE
# ==========================================
st.set_page_config(page_title="AI Sentiment Dashboard", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    .prediction-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #007bff; }
    </style>
    """, unsafe_allow_html=True)

LABEL_ORDER = ["Positive", "Neutral", "Negative"]

# ==========================================
# 2. CACHED MODELS (تحميل النماذج لمرة واحدة)
# ==========================================
@st.cache_resource
def load_models():
    text_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    image_pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    return text_pipe, image_pipe

text_engine, image_engine = load_models()

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_bert_prediction(text):
    try:
        cleaned = clean_text(text)
        if not cleaned: return "Neutral", 0.0
        result = text_engine(cleaned)[0]
        stars = int(result['label'].split()[0])
        score = result['score']
        if stars <= 2: label = "Negative"
        elif stars == 3: label = "Neutral"
        else: label = "Positive"
        return label, score
    except:
        return "Neutral", 0.0

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🤖 AI Control Panel")
st.sidebar.markdown("---")
mode = st.sidebar.radio("اختر وضع التحليل:", 
                        ["تحليل النصوص 📝", "تحليل الصور 🖼️", "تحليل الملفات 📂"])

st.sidebar.markdown("---")
st.sidebar.info("هذا التطبيق يستخدم نماذج BERT و ViT لتحليل المشاعر بدقة احترافية.")

# ==========================================
# 5. MODE: TEXT ANALYSIS
# ==========================================
if mode == "تحليل النصوص 📝":
    st.header("📝 تحليل مشاعر النصوص (BERT)")
    user_input = st.text_area("أدخل النص المراد تحليله هنا:", placeholder="أنا سعيد جداً باستخدام هذا التطبيق...")
    
    if st.button("تحليل النص"):
        if user_input:
            label, conf = get_bert_prediction(user_input)
            
            # عرض النتيجة في بطاقة
            st.markdown(f"""
            <div class="prediction-card">
                <h3>النتيجة: {label}</h3>
                <p>نسبة الثقة: <b>{conf:.2%}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # بروجرس بار للثقة
            st.progress(conf)
        else:
            st.warning("يرجى إدخال نص أولاً.")

# ==========================================
# 6. MODE: IMAGE ANALYSIS
# ==========================================
elif mode == "تحليل الصور 🖼️":
    st.header("🖼️ تحليل مشاعر الوجه (Vision Transformer)")
    uploaded_file = st.file_uploader("قم برفع صورة وجه...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="الصورة المرفوعة", use_container_width=True)
        
        with col2:
            with st.spinner('جاري تحليل الصورة...'):
                results = image_engine(img)
                st.subheader("نتائج التحليل:")
                for res in results[:3]:
                    st.write(f"**{res['label']}**: {res['score']:.2%}")
                    st.progress(res['score'])

# ==========================================
# 7. MODE: FILE ANALYSIS
# ==========================================
elif mode == "تحليل الملفات 📂":
    st.header("📂 تحليل الملفات والمقارنة الأكاديمية")
    uploaded_file = st.file_uploader("ارفع ملف Excel أو CSV", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
        st.write("معاينة البيانات:", df.head())
        
        col_text = st.selectbox("اختر عمود النصوص:", df.columns)
        col_label = st.selectbox("اختر عمود التصنيف الحقيقي (Ground Truth):", df.columns)
        
        if st.button("بدء المعالجة والمقارنة"):
            with st.spinner('جاري تشغيل النماذج (قد يستغرق وقتاً)...'):
                df = df.dropna(subset=[col_text, col_label]).reset_index(drop=True)

                # --- BERT Prediction ---
                bert_results = df[col_text].apply(get_bert_prediction)
                df['BERT_Label'] = [x[0] for x in bert_results]
                
                # --- ML Logic Regression ---
                X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                    df[col_text], df[col_label], df.index, test_size=0.2, random_state=42
                )
                
                vectorizer = TfidfVectorizer(preprocessor=clean_text, max_features=2500)
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
                
                lr_model = LogisticRegression(max_iter=1000)
                lr_model.fit(X_train_vec, y_train)
                
                # --- Visualizations ---
                st.subheader("📊 الرسوم البيانية")
                c1, c2 = st.columns(2)
                
                with c1:
                    # WordCloud
                    all_text = " ".join(df[col_text].astype(str))
                    wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    st.caption("سحابة الكلمات الأكثر تكراراً")

                with c2:
                    # Distribution
                    fig, ax = plt.subplots()
                    sns.countplot(data=df, x='BERT_Label', order=LABEL_ORDER, palette="viridis", ax=ax)
                    st.pyplot(fig)
                    st.caption("توزيع المشاعر (BERT)")

                # --- Metrics Comparison ---
                st.subheader("🏆 مقارنة أداء النماذج")
                y_test_bert = df.loc[idx_test, 'BERT_Label']
                y_test_lr = lr_model.predict(X_test_vec)
                
                acc_bert = accuracy_score(y_test, y_test_bert)
                acc_lr = accuracy_score(y_test, y_test_lr)
                
                m1, m2 = st.columns(2)
                m1.metric("دقة نموذج BERT", f"{acc_bert:.2%}")
                m2.metric("دقة Logistic Regression", f"{acc_lr:.2%}")

                # Confusion Matrices
                col_cm1, col_cm2 = st.columns(2)
                with col_cm1:
                    st.write("**BERT Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_test_bert, labels=LABEL_ORDER)
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay(cm, display_labels=LABEL_ORDER).plot(cmap='Blues', ax=ax)
                    st.pyplot(fig)
                
                with col_cm2:
                    st.write("**ML Confusion Matrix**")
                    cm_lr = confusion_matrix(y_test, y_test_lr, labels=LABEL_ORDER)
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay(cm_lr, display_labels=LABEL_ORDER).plot(cmap='Greens', ax=ax)
                    st.pyplot(fig)

                # Export Result
                st.subheader("📥 تحميل النتائج")
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                st.download_button(label="تحميل ملف النتائج النهائي", data=output.getvalue(), 
                                   file_name="sentiment_analysis_results.xlsx", 
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

