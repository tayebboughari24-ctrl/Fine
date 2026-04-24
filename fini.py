import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import torch
from datetime import datetime
import io
import time

# --- 1. إعدادات الهوية البصرية (SaaS UI/UX) ---
st.set_page_config(
    page_title="Sentify Pro | AI Sentiment Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تصميم عصري مخصص باستخدام CSS
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
    .stTextArea>div>div>textarea { border-radius: 15px; border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_status=True)

# --- 2. محرك الذكاء الاصطناعي (Backend Optimized) ---

@st.cache_resource(show_spinner="جاري تحميل محرك الذكاء الاصطناعي...")
def load_sentiment_pipeline():
    """تحميل النموذج مرة واحدة فقط مع دعم الـ GPU إذا توفر"""
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
        """تحليل النصوص بنظام الدفعات (Batch Processing) لسرعة قصوى"""
        return self.engine(texts, batch_size=16)

# --- 3. وظائف مساعدة ---
def init_state():
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None

def save_to_history(text, label, score):
    st.session_state.history.append({
        "الوقت": datetime.now().strftime("%H:%M:%S"),
        "النص": text[:50] + "...",
        "التصنيف": label,
        "الثقة": f"{score:.2%}"
    })

# --- 4. واجهة المستخدم الرسومية ---
def main():
    init_state()
    analyzer = SentimentAnalyzer()
    
    # القائمة الجانبية
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>Sentify Pro</h1>", unsafe_allow_status=True)
        st.image("https://cdn-icons-png.flaticon.com/512/8653/8653200.png", width=100)
        st.divider()
        
        mode = st.selectbox("🎯 اختر وضع التشغيل", ["التحليل السريع", "معالجة الملفات الكبيرة", "سجل العمليات"])
        
        st.divider()
        if st.button("مسح جميع البيانات 🗑️", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_analysis = None
            st.rerun()

    # --- الوضع الأول: التحليل السريع ---
    if mode == "التحليل السريع":
        st.title("🚀 تحليل المشاعر الفوري")
        st.markdown("أدخل أي نص باللغة الإنجليزية للحصول على تحليل دقيق ومعمق.")

        col_in, col_out = st.columns([1.2, 0.8], gap="large")

        with col_in:
            st.subheader("مدخلات النص")
            example = "This platform has completely transformed how we handle customer feedback. The UI is sleek and the speed is unmatched!"
            
            if st.button("✨ تجربة نص احترافي"):
                st.session_state.user_text = example
            
            user_input = st.text_area(
                "ماذا يدور في ذهنك؟",
                value=st.session_state.get('user_text', ""),
                height=200,
                placeholder="Paste content here..."
            )

            if st.button("بدء التحليل الذكي 🔍", type="primary", use_container_width=True):
                if user_input.strip():
                    with st.spinner("جاري المعالجة عبر الشبكة العصبية..."):
                        start_time = time.time()
                        res = analyzer.analyze([user_input])[0]
                        duration = time.time() - start_time
                        
                        st.session_state.last_analysis = {**res, "duration": duration}
                        save_to_history(user_input, res['label'], res['score'])
                else:
                    st.warning("الرجاء كتابة نص أولاً.")

        with col_out:
            st.subheader("النتيجة التحليلية")
            if st.session_state.last_analysis:
                data = st.session_state.last_analysis
                color = "#10b981" if data['label'] == "POSITIVE" else "#ef4444"
                
                st.markdown(f"""
                <div class="sentiment-card" style="border-color: {color};">
                    <h1 style="color: {color}; margin: 0;">{data['label']}</h1>
                    <p style="color: #64748b; font-size: 1.1em;">نسبة الثقة: {data['score']:.2%}</p>
                    <small>وقت التنفيذ: {data['duration']:.3f} ثانية</small>
                </div>
                """, unsafe_allow_status=True)

                # رسم بياني دائري صغير
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = data['score'] * 100,
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': color}}
                ))
                fig.update_layout(height=250, margin=dict(t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ستظهر نتائج التحليل هنا بمجرد البدء.")

    # --- الوضع الثاني: معالجة الملفات ---
    elif mode == "معالجة الملفات الكبيرة":
        st.title("📂 المعالجة الجماعية (Bulk Processing)")
        st.markdown("ارفع ملفات CSV تحتوي على آلاف التعليقات لتحليلها دفعة واحدة.")
        
        uploaded_file = st.file_uploader("اختر ملف (CSV / Excel)", type=["csv", "xlsx"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
            st.success(f"تم تحميل الملف بنجاح! يحتوي على {len(df)} صف.")
            
            target_col = st.selectbox("اختر العمود الذي يحتوي على النصوص:", df.columns)
            
            if st.button("بدء المعالجة الشاملة 🚀", type="primary"):
                with st.status("جاري تحليل البيانات الضخمة...") as status:
                    texts = df[target_col].astype(str).tolist()
                    results = analyzer.analyze(texts)
                    
                    df['Sentiment'] = [r['label'] for r in results]
                    df['Confidence'] = [r['score'] for r in results]
                    status.update(label="اكتمل التحليل!", state="complete")

                # عرض داشبورد مصغر للملف
                st.divider()
                c1, c2, c3 = st.columns(3)
                pos_count = len(df[df['Sentiment'] == 'POSITIVE'])
                c1.metric("إجمالي النصوص", len(df))
                c2.metric("إيجابي ✅", pos_count)
                c3.metric("سلبي ❌", len(df) - pos_count)

                # الرسم البياني للملف
                fig_bar = px.histogram(df, x="Sentiment", color="Sentiment", 
                                      color_discrete_map={'POSITIVE': '#10b981', 'NEGATIVE': '#ef4444'})
                st.plotly_chart(fig_bar, use_container_width=True)

                # تحميل النتائج
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button("تحميل تقرير النتائج (CSV) ⬇️", data=csv_data, file_name="analysis_report.csv", use_container_width=True)

    # --- الوضع الثالث: السجل ---
    elif mode == "سجل العمليات":
        st.title("📜 سجل النشاط")
        if st.session_state.history:
            df_history = pd.DataFrame(st.session_state.history)
            st.table(df_history)
            
            # تحليل بسيط للسجل
            if len(df_history) > 1:
                fig_hist = px.pie(df_history, names="التصنيف", title="توزيع المشاعر في السجل الحالي")
                st.plotly_chart(fig_hist)
        else:
            st.info("السجل فارغ حالياً.")

if __name__ == "__main__":
    main()
        
