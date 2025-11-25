import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
import altair as alt
import graphviz
import io
import datetime

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Super Novel Analyst (Dashboard)", page_icon="📊", layout="wide")

# --- 1. ระบบความจำ (Session State) ---
# สร้างสมุดจดประวัติ ถ้ายังไม่มีให้สร้างใหม่
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("📊 Super Novel Analyst: พร้อมระบบ Dashboard")
st.caption("วิเคราะห์นิยาย + บันทึกประวัติการใช้งาน (อย่าลืมกดดาวน์โหลดก่อนปิดหน้านะครับ!)")

# --- Sidebar ---
with st.sidebar:
    st.header("🔑 1. ตั้งค่า")
    api_key = st.text_input("Google API Key:", type="password")
    
    st.divider()
    st.header("🤖 2. โมเดล AI")
    
    # ตัวเลือกโมเดลแบบ Manual (เพื่อความชัวร์)
    model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
    
    # พยายามดึงโมเดลจริง ถ้า Key ถูกต้อง
    if api_key:
        try:
            genai.configure(api_key=api_key)
            real_models = []
            for m in genai.list_models():
                if 'gemini' in m.name and 'generateContent' in m.supported_generation_methods:
                    real_models.append(m.name)
            if real_models:
                real_models.sort(reverse=True)
                model_options = real_models
        except:
            pass
            
    selected_model_name = st.selectbox("เลือกเวอร์ชั่น:", model_options)

    st.divider()
    st.info(f"📚 วิเคราะห์ไปแล้ว: {len(st.session_state.history)} เรื่อง")

# --- ส่วนแสดงผลหลัก (Main Area) ---
# สร้าง Tabs ใหญ่ เพื่อแยกหน้าวิเคราะห์ กับ หน้า Dashboard
main_tab1, main_tab2 = st.tabs(["🕵️‍♀️ วิเคราะห์นิยาย", "🏆 Dashboard & ประวัติ"])

# ==========================================
# TAB 1: หน้าวิเคราะห์ (เหมือนเดิม)
# ==========================================
with main_tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("อัปโหลดไฟล์")
        uploaded_file = st.file_uploader("เลือกไฟล์นิยาย (.txt)", type=['txt'])
        novel_text = ""
        if uploaded_file:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            novel_text = stringio.read()
            st.success(f"✅ อ่านไฟล์สำเร็จ ({len(novel_text):,} ตัวอักษร)")

    with col2:
        st.subheader("สั่งการ AI")
        if not api_key:
            st.warning("⚠️ ใส่ API Key ก่อนครับ")
            analyze_btn = False
        else:
            short_name = selected_model_name.split('/')[-1]
            analyze_btn = st.button(f"🚀 ส่งให้ {short_name} วิเคราะห์", type="primary", use_container_width=True)

    if analyze_btn and novel_text and api_key:
        # Prompt
        prompt = f"""
        Analyze this novel. Return ONLY JSON.
        Language: THAI.
        Structure:
        {{
          "title": "ตั้งชื่อเรื่องให้นิยายนี้ (เดาจากเนื้อหา)",
          "summary": "เรื่องย่อ (3 บรรทัด)",
          "genre": "แนวเรื่อง",
          "characters": [{{"name": "ชื่อ", "role": "บทบาท"}}],
          "relations": [{{"source": "A", "target": "B", "relation": "ความสัมพันธ์", "weight": 1-10}}],
          "sentiment_arc": [{{"chapter_part": 1, "score": 10, "mood": "สุข"}}],
          "critique": {{ "strengths": [], "weaknesses": [], "plot_holes": [] }},
          "overall_score": 8.5 (คะแนนภาพรวม 0-10)
        }}
        Text: {novel_text[:800000]}
        """

        with st.spinner(f'⚡ {short_name} กำลังทำงาน...'):
            try:
                model = genai.GenerativeModel(selected_model_name, generation_config={"response_mime_type": "application/json"})
                response = model.generate_content(prompt)
                
                # Clean JSON
                json_str = response.text.replace("```json", "").replace("```", "").strip()
                data = json.loads(json_str)
                
                st.toast("วิเคราะห์สำเร็จ! บันทึกลง Dashboard แล้ว", icon="💾")
                
                # --- บันทึกลงสมุดประวัติ (History) ---
                record = {
                    "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                    "Title": data.get('title', 'Unknown'),
                    "Genre": data.get('genre', '-'),
                    "Score": data.get('overall_score', 0),
                    "Characters": len(data.get('characters', [])),
                    "Summary": data.get('summary', ''),
                    "Model": short_name
                }
                st.session_state.history.append(record)
                
                # --- แสดงผลลัพธ์ปัจจุบัน ---
                t1, t2, t3 = st.tabs(["📝 ผลวิเคราะห์", "🕸️ ความสัมพันธ์", "📈 กราฟอารมณ์"])
                
                with t1:
                    st.header(f"เรื่อง: {data.get('title')}")
                    st.info(data.get('summary'))
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("✅ **จุดแข็ง**")
                        for x in data.get('critique', {}).get('strengths', []): st.write(f"- {x}")
                    with c2:
                        st.write("❌ **จุดอ่อน**")
                        for x in data.get('critique', {}).get('weaknesses', []): st.write(f"- {x}")
                        
                with t2:
                    graph = graphviz.Digraph(graph_attr={'rankdir':'LR'})
                    for r in data.get('relations', []):
                        graph.edge(r.get('source','?'), r.get('target','?'), label=r.get('relation',''), penwidth=str(r.get('weight',1)/2))
                    st.graphviz_chart(graph)
                    
                with t3:
                    df = pd.DataFrame(data.get('sentiment_arc', []))
                    if not df.empty:
                        c = alt.Chart(df).mark_line(point=True).encode(x='chapter_part', y='score', tooltip=['mood'], color=alt.value('#FF4B4B')).interactive()
                        st.altair_chart(c, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# TAB 2: Dashboard & ประวัติ (ของใหม่!)
# ==========================================
with main_tab2:
    st.header("🏆 ประวัติการวิเคราะห์ (Session History)")
    
    if len(st.session_state.history) > 0:
        # แปลงข้อมูลเป็นตาราง
        df_history = pd.DataFrame(st.session_state.history)
        
        # 1. แสดงตารางข้อมูล
        st.dataframe(df_history, use_container_width=True)
        
        # 2. กราฟเปรียบเทียบคะแนน
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 เปรียบเทียบคะแนนแต่ละเรื่อง")
            bar_chart = alt.Chart(df_history).mark_bar().encode(
                x=alt.X('Title', title='ชื่อเรื่อง'),
                y=alt.Y('Score', title='คะแนน (0-10)'),
                color='Genre',
                tooltip=['Title', 'Genre', 'Score']
            )
            st.altair_chart(bar_chart, use_container_width=True)
            
        with c2:
            st.subheader("🍰 สัดส่วนแนวประเภทยิยาย")
            pie_chart = alt.Chart(df_history).mark_arc().encode(
                theta=alt.Theta("count()"),
                color="Genre",
                tooltip=["Genre", "count()"]
            )
            st.altair_chart(pie_chart, use_container_width=True)

        # 3. ปุ่มดาวน์โหลด
        st.divider()
        st.subheader("💾 เก็บข้อมูลกลับบ้าน")
        
        # แปลงเป็น CSV
        csv = df_history.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 ดาวน์โหลดประวัติเป็นไฟล์ CSV (เปิดใน Excel ได้)",
            data=csv,
            file_name='novel_analysis_history.csv',
            mime='text/csv',
            type="primary"
        )
        st.caption("*คำเตือน: ถ้าปิดเว็บนี้ ข้อมูลประวัติจะหายไป อย่าลืมกดดาวน์โหลดนะครับ")
        
    else:
        st.info("📭 ยังไม่มีข้อมูลประวัติ ลองไปที่แท็บ 'วิเคราะห์นิยาย' แล้วเริ่มวิเคราะห์เรื่องแรกเลย!")
