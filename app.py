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

# --- 1. สร้างตัวแปรความจำ (Session State) ---
# เพื่อแก้ปัญหาหน้าจอหายหลังวิเคราะห์เสร็จ
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'result_data' not in st.session_state:
    st.session_state.result_data = {}

st.title("🤖 Pro Novel Analyst: ระบบวิเคราะห์นิยายอัจฉริยะ")

# --- ส่วนรับข้อมูล ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. ใส่เนื้อหานิยาย")
    input_method = st.radio("เลือกวิธีนำเข้าข้อมูล:", ["📂 อัปโหลดไฟล์ (.txt)", "✍️ วางข้อความเอง"])
    
    novel_text = ""
    
    if input_method == "📂 อัปโหลดไฟล์ (.txt)":
        uploaded_file = st.file_uploader("เลือกไฟล์นิยายของคุณ", type=['txt'])
        if uploaded_file is not None:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            novel_text = stringio.read()
            st.success(f"✅ อ่านไฟล์สำเร็จ! (ความยาว {len(novel_text):,} ตัวอักษร)")
            
    else:
        novel_text = st.text_area("วางเนื้อหานิยายที่นี่:", height=300)

with col_right:
    st.subheader("2. ตั้งค่าการวิเคราะห์")
    manual_chars = st.text_area("เพิ่มชื่อตัวละครเอง (คั่นด้วยจุลภาค)", placeholder="เช่น: สมชาย, สมหญิง", height=100)
    
    # ปุ่มกดเปลี่ยนสถานะความจำ
    if st.button("🚀 สั่ง AI วิเคราะห์เดี๋ยวนี้!", type="primary", use_container_width=True):
        if novel_text:
            st.session_state.analyzed = True
            st.rerun() # สั่งรีเฟรชหน้าจอเพื่อแสดงผลทันที
        else:
            st.error("⚠️ กรุณาใส่เนื้อหานิยายก่อนครับ")

# --- ฟังก์ชัน AI (NER Engine) ---
@st.cache_resource
def load_ner_engine():
    return NER("thainer")

def extract_entities(text):
    ner = load_ner_engine()
    # ตัดข้อความให้สั้นลงหน่อยถ้ายาวเกินไป (ป้องกันรอนานเกิน 5 นาที)
    # แต่ถ้าเครื่องไหวก็เอา limit ออกได้
    processed_text = text[:100000] if len(text) > 100000 else text 
    tags = ner.tag(processed_text)
    
    entities = {
        "PERSON": [],
        "LOCATION": [],
        "DATE": [],
        "TIME": []
    }
    
    # คำสรรพนามที่ AI มักเข้าใจผิดว่าเป็นชื่อคน (Blacklist)
    blacklist_names = [
        "เขา", "เธอ", "มัน", "ฉัน", "ผม", "กู", "มึง", "ข้า", "เอ็ง", "เรา", "พวกเรา",
        "พี่", "น้อง", "ลุง", "ป้า", "น้า", "อา", "พ่อ", "แม่", "ปู่", "ย่า", "ตา", "ยาย",
        "คุณ", "ท่าน", "แก", "ใคร", "นาง", "นาย", "เด็ก", "ผู้ชาย", "ผู้หญิง", "คน",
        "ตัวเอง", "บ่าว", "ฝ่าบาท", "พระองค์", "หมอ", "ครู", "อาจารย์"
    ]

    for item in tags:
        if len(item) == 3:
            word, pos, tag = item
        elif len(item) == 2:
            word, tag = item
        else:
            continue 
            
        word_clean = word.strip()
        
        if "PERSON" in tag:
            # กรองคำสั้นเกิน 1 ตัวอักษร และคำที่อยู่ใน Blacklist
            if len(word_clean) > 1 and word_clean not in blacklist_names:
                entities["PERSON"].append(word_clean)
        elif "LOCATION" in tag:
            entities["LOCATION"].append(word_clean)
        elif "DATE" in tag:
            entities["DATE"].append(word_clean)
        elif "TIME" in tag:
            entities["TIME"].append(word_clean)
            
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

# --- ส่วนแสดงผล (ทำงานเมื่อสถานะ analyzed เป็น True) ---
if st.session_state.analyzed and novel_text:
    
    st.divider()
    
    with st.spinner('🤖 AI กำลังทำงาน... (กรองชื่อซ้ำ + สร้างกราฟ)'):
        
        # 1. ตัดคำ
        raw_words = word_tokenize(novel_text, engine="newmm")
        words = [w for w in raw_words if w.strip() != "" and isthai(w)]
        
        # 2. เรียก AI หาชื่อ
        found_entities = extract_entities(novel_text)
        
        # 3. รวมชื่อและจัดการคำซ้ำ (ใช้ set เพื่อเอาเฉพาะชื่อที่ไม่ซ้ำกัน)
        auto_chars = list(set(found_entities["PERSON"])) 
        user_chars = [c.strip() for c in manual_chars.split(",") if c.strip() != ""]
        
        # รวม + ตัดซ้ำอีกรอบ
        final_char_list = list(set(auto_chars + user_chars))
        final_char_list.sort() # เรียงตามตัวอักษร

        st.success(f"✅ วิเคราะห์เสร็จสิ้น! พบตัวละคร (Unique) ทั้งหมด {len(final_char_list)} คน")

        # --- สร้าง Tabs ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ สิ่งที่ AI พบ
