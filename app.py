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
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ สิ่งที่ AI พบ", "📊 สถิติพื้นฐาน", "📈 กราฟอารมณ์", "🕸️ ความสัมพันธ์", "📝 คำซ้ำ"])

        # TAB 1: AI Findings
        with tab1:
            st.header("🔍 สิ่งที่ AI ค้นเจอ")
            st.info("💡 หมายเหตุ: ระบบกรองคำสรรพนาม (เขา, เธอ, ฉัน) ออกให้แล้ว")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(f"**👤 ตัวละคร ({len(final_char_list)})**")
                # แสดงเป็น Chip สวยๆ
                st.write(", ".join(final_char_list))
            with c2:
                locs = list(set(found_entities['LOCATION']))
                st.write(f"**📍 สถานที่ ({len(locs)})**")
                st.write(", ".join(locs))
            with c3:
                times = list(set(found_entities['DATE'] + found_entities['TIME']))
                st.write(f"**📅 เวลา ({len(times)})**")
                st.write(", ".join(times))

        # TAB 2: Stats
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

        # TAB 3: Sentiment
        with tab3:
            st.header("กราฟอารมณ์")
            chunk_size = 500 # เพิ่มขนาด chunk ให้อ่านกราฟง่ายขึ้น
            chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
            sentiment_scores = [analyze_sentiment(chunk) for chunk in chunks]
            chart_data = pd.DataFrame({'Position': range(len(sentiment_scores)), 'Score': sentiment_scores})
            line_chart = alt.Chart(chart_data).mark_line(interpolate='basis').encode(
                x=alt.X('Position', title='ช่วงเวลาของเรื่อง'),
                y=alt.Y('Score', title='คะแนนอารมณ์'),
                color=alt.value("#FF4B4B"),
                tooltip=['Position', 'Score']
            ).properties(height=350)
            st.altair_chart(line_chart, use_container_width=True)

        # TAB 4: Network
        with tab4:
            st.header("เครือข่ายความสัมพันธ์")
            if not final_char_list:
                st.warning("ไม่พบตัวละคร")
            else:
                # เลือกเฉพาะตัวละคร Top 15 ตัวแรกที่เจอบ่อยสุด เพื่อไม่ให้กราฟรกจนดูไม่รู้เรื่อง
                # นับความถี่ชื่อในเนื้อหา
                char_freq = {name: novel_text.count(name) for name in final_char_list}
                # เรียงลำดับเอาเฉพาะ Top 15
                top_chars = sorted(char_freq, key=char_freq.get, reverse=True)[:15]
                
                graph = graphviz.Digraph()
                graph.attr(rankdir='LR')
                
                # Logic เดิม แต่ใช้ Top Chars
                paragraphs = novel_text.split('\n')
                relations = Counter()
                
                for para in paragraphs:
                    # หาเฉพาะตัวละคร Top 15 ในย่อหน้านั้น
                    found_in_para = [c for c in top_chars if c in para]
                    if len(found_in_para) > 1:
                        for i in range(len(found_in_para)):
                            for j in range(i+1, len(found_in_para)):
                                pair = tuple(sorted([found_in_para[i], found_in_para[j]]))
                                relations[pair] += 1
                                
                # วาดกราฟ
                for char in top_chars:
                    graph.node(char, style='filled', fillcolor='#D3D3D3')
                    
                for (char1, char2), weight in relations.items():
                    if weight > 0:
                        # ปรับขนาดเส้นไม่ให้หนาเกินไป
                        pen_width = min(weight/2, 5) 
                        graph.edge(char1, char2, penwidth=str(pen_width), label=str(weight))
                        
                st.graphviz_chart(graph)
                st.caption(f"แสดงเฉพาะตัวละครหลัก 15 ตัวแรก (จากทั้งหมด {len(final_char_list)} ตัว) เพื่อความสวยงาม")

        # TAB 5: Word Cloud
        with tab5:
            st.header("คำที่ใช้บ่อย")
            word_counts = Counter(words)
            df_words = pd.DataFrame(word_counts.most_common(20), columns=['คำศัพท์', 'จำนวนครั้ง'])
            st.dataframe(df_words, use_container_width=True)
