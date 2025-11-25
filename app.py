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
    
    # --- จุดที่แก้ไข: เช็คจำนวนค่าที่ส่งกลับมาก่อนดึงข้อมูล ---
    for item in tags:
        # รับค่าแบบยืดหยุ่น (ป้องกัน Error)
        if len(item) == 3:
            word, pos, tag = item
        elif len(item) == 2:
            word, tag = item
        else:
            continue # ถ้ามาแปลกๆ ให้ข้ามไป
            
        # เก็บข้อมูลตาม Tag
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
    # แก้ไขบรรทัดนี้ที่ error ครับ (เติมเครื่องหมายปิดให้ครบ)
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

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ สิ่งที่ AI พบ", "📊 สถิติพื้นฐาน", "📈 กราฟอารมณ์", "🕸️ ความสัมพันธ์", "📝 คำซ้ำ"])

        # TAB 1
        with tab1:
            st.header("🔍 สิ่งที่ AI ค้นเจอ")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.info(f"👤 ตัวละคร ({len(set(found_entities['PERSON']))})")
                st.write(", ".join(set(found_entities['PERSON'])))
            with c2:
                st.warning(f"📍 สถานที่ ({len(set(found_entities['LOCATION']))})")
                st.write(", ".join(set(found_entities['LOCATION'])))
            with c3:
                st.success(f"📅 เวลา ({len(set(found_entities['DATE'] + found_entities['TIME']))})")
                st.write(", ".join(set(found_entities['DATE'] + found_entities['TIME'])))

        # TAB 2
        with tab2:
            st.header("สถิติภาพรวม")
            n_words = len(words)
            read_time = round(n_words / 200)
            vocab = set(words)
            diversity = round((len(vocab) / n_words) * 100, 2)
            c1, c2, c3 = st.columns(3)
            c1.metric("จำนวนคำทั้งหมด", f"{n_words:,}")
            c2.metric("เวลาอ่าน (นาที)", read_time)
            c3.metric("ความหลากหลายคำ", f"{diversity}%")

        # TAB 3
        with tab3:
            st.header("กราฟอารมณ์")
            chunk_size = 100
            chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
            sentiment_scores = [analyze_sentiment(chunk) for chunk in chunks]
            chart_data = pd.DataFrame({'Position': range(len(sentiment_scores)), 'Score': sentiment_scores})
            line_chart = alt.Chart(chart_data).mark_line(interpolate='basis').encode(
                x='Position', y='Score', color=alt.value("#FF4B4B")
            ).properties(height=300)
            st.altair_chart(line_chart, use_container_width=True)

        # TAB 4
        with tab4:
            st.header("เครือข่ายความสัมพันธ์")
            if not final_char_list:
                st.warning("ไม่พบตัวละคร ลองพิมพ์เพิ่มเอง")
            else:
                graph = graphviz.Digraph()
                graph.attr(rankdir='LR')
                paragraphs = novel_text.split('\n')
                relations = Counter()
                for para in paragraphs:
                    found_in_para = [c for c in final_char_list if c in para]
                    if len(found_in_para) > 1:
                        for i in range(len(found_in_para)):
                            for j in range(i+1, len(found_in_para)):
                                pair = tuple(sorted([found_in_para[i], found_in_para[j]]))
                                relations[pair] += 1
                for (char1, char2), weight in relations.items():
                    if weight > 0:
                        graph.edge(char1, char2, penwidth=str(weight/2), label=str(weight))
                        graph.node(char1, style='filled', fillcolor='#D3D3D3')
                        graph.node(char2, style='filled', fillcolor='#D3D3D3')
                st.graphviz_chart(graph)

        # TAB 5
        with tab5:
            st.header("คำที่ใช้บ่อย")
            word_counts = Counter(words)
            df_words = pd.DataFrame(word_counts.most_common(20), columns=['คำศัพท์', 'จำนวนครั้ง'])
            st.dataframe(df_words, use_
