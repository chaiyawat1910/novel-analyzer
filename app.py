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
            st.success(f"✅ อ่านไฟล์สำเร็จ! (ความยาว {len(novel_text)} ตัวอักษร)")
            
    else:
        novel_text = st.text_area("วางเนื้อหานิยายที่นี่:", height=300)

with col_right:
    st.subheader("2. ตั้งค่าการวิเคราะห์")
    manual_chars = st.text_area("เพิ่มชื่อตัวละครเอง (คั่นด้วยจุลภาค , )", placeholder="เช่น: สมชาย, สมหญิง", height=100)
    analyze_btn = st.button("🚀 สั่ง AI วิเคราะห์เดี๋ยวนี้!", type="primary", use_container_width=True)

# --- ฟังก์ชัน AI (NER Engine) ---
@st.cache_resource
def load_ner_engine():
    return NER("thainer")

def extract_entities(text):
    ner = load_ner_engine()
    tags = ner.tag(text)
    
    entities = {
        "PERSON": [],
        "LOCATION": [],
        "DATE": [],
        "TIME": []
    }
    
    for item in tags:
        # รับค่าแบบยืดหยุ่น (ป้องกัน Error ValueError)
        if len(item) == 3:
            word, pos, tag = item
        elif len(item) == 2:
            word, tag = item
        else:
            continue 
            
        if "PERSON" in tag:
            entities["PERSON"].append(word)
        elif "LOCATION" in tag:
            entities["LOCATION"].append(word)
        elif "DATE" in tag:
            entities["DATE"].append(word)
        elif "TIME" in tag:
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
        
        # 1. ตัดคำ
        raw_words = word_tokenize(novel_text, engine="newmm")
        words = [w for w in raw_words if w.strip() != "" and isthai(w)]
        
        # 2. เรียก AI หาชื่อ
        found_entities = extract_entities(novel_text)
        
        # รวมชื่อ
        auto_chars = list(set(found_entities["PERSON"]))
        user_chars = [c.strip() for c in manual_chars.split(",") if c.strip() != ""]
        final_char_list = list(set(auto_chars + user_chars))
        final_char_list = [c for c in final_char_list if len(c) > 1]

        # --- แสดงผล ---
        st.divider()
        st.success(f"✅ วิเคราะห์เสร็จสิ้น! พบตัวละครทั้งหมด {len(final_char_list)} คน")
