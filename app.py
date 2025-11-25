import streamlit as st
import pandas as pd
import altair as alt
from pythainlp import word_tokenize
from pythainlp.util import isthai
from pythainlp.tag import NER
from collections import Counter
import graphviz
import io

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Pro Novel Analyst AI", page_icon="🤖", layout="wide")

st.title("🤖 Pro Novel Analyst: ระบบวิเคราะห์นิยายอัจฉริยะ")
st.info("อัปเกรดใหม่! รองรับการอัปโหลดไฟล์ และค้นหาตัวละคร/สถานที่ให้เองอัตโนมัติ")

# --- ส่วนรับข้อมูล (อัปเกรดใหม่) ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. ใส่เนื้อหานิยาย")
    # Tab เลือกวิธีการใส่ข้อมูล
    input_method = st.radio("เลือกวิธีนำเข้าข้อมูล:", ["📂 อัปโหลดไฟล์ (.txt)", "✍️ วางข้อความเอง"])
    
    novel_text = ""
    
    if input_method == "📂 อัปโหลดไฟล์ (.txt)":
        uploaded_file = st.file_uploader("เลือกไฟล์นิยายของคุณ", type=['txt'])
        if uploaded_file is not None:
            # อ่านไฟล์และแปลงเป็นตัวหนังสือ
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            novel_text = stringio.read()
            st.success(f"✅ อ่านไฟล์สำเร็จ! (ความยาว {len(novel_text)} ตัวอักษร)")
            
    else:
        novel_text = st.text_area("วางเนื้อหานิยายที่นี่:", height=300)

with col_right:
    st.subheader("2. ตั้งค่าการวิเคราะห์")
    st.write("ระบบจะใช้ AI ค้นหาชื่อตัวละครให้เอง แต่ถ้าคุณต้องการเพิ่มชื่อเฉพาะเจาะจง สามารถพิมพ์เพิ่มได้ด้านล่าง")
    
    # รับชื่อเพิ่มเติม
    manual_chars = st.text_area("เพิ่มชื่อตัวละครเอง (คั่นด้วยจุลภาค , )", placeholder="เช่น: สมชาย, สมหญิง (ระบบจะรวมกับที่ AI หาเจอ)", height=100)
    
    # ปุ่มกด
    analyze_btn = st.button("🚀 สั่ง AI วิเคราะห์เดี๋ยวนี้!", type="primary", use_container_width=True)

# --- ฟังก์ชัน AI (NER Engine) ---
@st.cache_resource # เก็บแคชโมเดลไว้ จะได้ไม่ต้องโหลดใหม่ทุกครั้งให้เสียเวลา
def load_ner_engine():
    return NER("thainer")

def extract_entities(text):
    ner = load_ner_engine()
    # tag ผลลัพธ์จะเป็น list ของ (คำ, ชนิดคำ, ชนิด Entity)
    tags = ner.tag(text)
    
    entities = {
        "PERSON": [],
        "LOCATION": [],
        "DATE": [],
        "TIME": []
    }
    
    for word, pos, tag in tags:
        if tag == "B-PERSON" or tag == "I-PERSON":
            entities["PERSON"].append(word)
        elif tag == "B-LOCATION" or tag == "I-LOCATION":
            entities["LOCATION"].append(word)
        elif tag == "B-DATE" or tag == "I-DATE":
            entities["DATE"].append(word)
        elif tag == "B-TIME" or tag == "I-TIME":
            entities["TIME"].append(word)
            
    return entities

def analyze_sentiment(words):
    pos_words = ["รัก", "ดี", "สุข", "สวย", "ยิ้ม", "หัวเราะ", "ชอบ", "อบอุ่น", "หวาน", "ตื่นเต้น", "สำเร็จ", "รอด", "ชนะ"]
    neg_words = ["เกลียด", "ตาย", "ฆ่า", "เลว", "ร้องไห้", "เจ็บ", "โกรธ", "เศร้า", "ทรมาน", "กลัว", "มืดมน", "แพ้", "เจ็บปวด"]
    score = 0
    if len(words) > 0:
        pos_cnt = sum(1 for w in words if w in pos_words)
        neg_cnt = sum(1 for w in words if w in neg_words)
        score = pos_cnt - neg_cnt
    return score

# --- เริ่มวิเคราะห์เมื่อกดปุ่ม ---
if analyze_btn and novel_text:
    with st.spinner('🤖 AI กำลังทำงาน... (อ่านนิยาย > หาตัวละคร > สร้างกราฟ)'):
        
        # 1. ตัดคำ (Tokenization)
        raw_words = word_tokenize(novel_text, engine="newmm")
        words = [w for w in raw_words if w.strip() != "" and isthai(w)]
        
        # 2. เรียก AI หาชื่อ (NER Extraction)
        # ตัดข้อความมาแค่ 5000 ตัวแรกเพื่อหาชื่อก่อน (เพื่อความเร็ว) หรือจะส่งทั้งหมดก็ได้ถ้ารอไหว
        # ในที่นี้ส่งทั้งหมดแต่อาจจะช้าหน่อยสำหรับนิยายยาวมากๆ
        found_entities = extract_entities(novel_text)
        
        # รวมชื่อที่ AI เจอ + ชื่อที่คนพิมพ์เพิ่ม
        auto_chars = list(set(found_entities["PERSON"])) # ตัดคำซ้ำ
        user_chars = [c.strip() for c in manual_chars.split(",") if c.strip() != ""]
        final_char_list = list(set(auto_chars + user_chars))
        
        # กรองชื่อสั้นๆ ทิ้ง (เช่น "นา" "ก") ป้องกันขยะ
        final_char_list = [c for c in final_char_list if len(c) > 1]

        # --- ส่วนแสดงผล ---
        st.divider()
        st.success(f"✅ วิเคราะห์เสร็จสิ้น! พบตัวละครทั้งหมด {len(final_char_list)} คน")

        # สร้าง Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ สิ่งที่ AI พบ", "📊 สถิติพื้นฐาน", "📈 กราฟอารมณ์", "🕸️ ความสัมพันธ์", "📝 คำซ้ำ"])

        # === TAB 1: AI Discovery ===
        with tab1:
            st.header("🔍 สิ่งที่ AI ค้นเจอในเรื่อง (Named Entities)")
            col_ent1, col_ent2, col_ent3 = st.columns(3)
            
            with col_ent1:
                st.info(f"👤 ตัวละคร/ชื่อคน ({len(set(found_entities['PERSON']))})")
                st.write(", ".join(set(found_entities["PERSON"])))
                
            with col_ent2:
                st.warning(f"📍 สถานที่ ({len(set(found_entities['LOCATION']))})")
                st.write(", ".join(set(found_entities["LOCATION"])))
                
            with col_ent3:
                st.success(f"📅 วันและเวลา ({len(set(found_entities['DATE'] + found_entities['TIME']))})")
                st.write(", ".join(set(found_entities['DATE'] + found_entities['TIME'])))

        # === TAB 2: Basic Stats ===
        with tab2:
            st.header("สถิติภาพรวม")
            n_words = len(words)
            read_time = round(n_words / 200)
            vocab = set(words)
            diversity = round((len(vocab) / n_words) * 100, 2)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("จำนวนคำทั้งหมด", f"{n_words:,}")
