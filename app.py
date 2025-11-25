import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
import altair as alt
import graphviz
import io

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Super Novel Analyst (Latest AI)", page_icon="🚀", layout="wide")

st.title("🚀 Super Novel Analyst: เลือกใช้ AI รุ่นล่าสุดได้เอง")
st.caption("ระบบดึงรายชื่อโมเดล Real-time: รองรับ Gemini 1.5 / 2.0 / Next Gen")

# --- Sidebar ---
with st.sidebar:
    st.header("🔑 1. ตั้งค่ากุญแจ (API Key)")
    api_key = st.text_input("วาง Google API Key:", type="password")
    st.markdown("[กดขอ API Key ฟรีที่นี่](https://aistudio.google.com/app/apikey)")
    
    st.divider()
    
    st.header("🤖 2. เลือกโมเดล AI")
    selected_model_name = None
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # ดึงรายชื่อโมเดลทั้งหมดที่มีสิทธิ์ใช้
            model_list = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    # เอาเฉพาะตระกูล Gemini
                    if 'gemini' in m.name:
                        model_list.append(m.name)
            
            # เรียงลำดับเอาตัวใหม่ๆ ขึ้นก่อน (เรียงตามชื่อ)
            model_list.sort(reverse=True)
            
            if model_list:
                selected_model_name = st.selectbox(
                    "เลือกเวอร์ชั่นโมเดล:", 
                    model_list, 
                    index=0 # เลือกตัวแรกสุด (มักจะเป็นตัวใหม่สุด)
                )
                st.success(f"กำลังใช้: {selected_model_name}")
            else:
                st.error("ไม่พบโมเดล Gemini ใน Key นี้")
        except Exception as e:
            st.error(f"API Key ผิดพลาด หรือ เชื่อมต่อไม่ได้: {e}")

# --- ส่วนรับข้อมูล ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("3. อัปโหลดนิยาย")
    uploaded_file = st.file_uploader("เลือกไฟล์นิยาย (.txt)", type=['txt'])
    novel_text = ""
    if uploaded_file:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        novel_text = stringio.read()
        st.success(f"✅ อ่านไฟล์สำเร็จ ({len(novel_text):,} ตัวอักษร)")

with col2:
    st.subheader("4. สั่งการ")
    if not api_key or not selected_model_name:
        st.warning("⚠️ กรุณาใส่ Key และเลือกโมเดลทางซ้ายมือก่อน")
        analyze_btn = False
    else:
        analyze_btn = st.button(f"🚀 ส่งให้ {selected_model_name.split('/')[-1]} วิเคราะห์!", type="primary", use_container_width=True)

# --- Logic การทำงาน ---
if analyze_btn and novel_text and api_key and selected_model_name:
    
    # Prompt สำหรับสั่งงาน
    prompt = f"""
    Analyze this novel text and return ONLY JSON format.
    Role: Professional Literature Critic.
    Language: THAI (ภาษาไทย).
    
    Structure:
    {{
      "summary": "เรื่องย่อ (5 บรรทัด)",
      "genre": "แนวเรื่อง",
      "characters": [{{"name": "ชื่อ", "role": "บทบาท", "traits": "นิสัย"}}],
      "relations": [{{"source": "A", "target": "B", "relation": "ความสัมพันธ์", "weight": 1-10}}],
      "sentiment_arc": [{{"chapter_part": 1, "score": 10, "mood": "สุข"}}],
      "critique": {{ "strengths": [], "weaknesses": [], "plot_holes": [] }}
    }}
    *For sentiment_arc, split story into 10 parts, score -10 to 10.
    
    NO markdown code blocks (```json). Just raw JSON string.
    
    Text:
    {novel_text[:800000]} 
    """

    with st.spinner(f'⚡ {selected_model_name} กำลังอ่านนิยาย...'):
        try:
            # เรียกใช้โมเดลตามที่ผู้ใช้เลือก
            model = genai.GenerativeModel(
                selected_model_name, 
                generation_config={"response_mime_type": "application/json"}
            )
            
            response = model.generate_content(prompt)
            
            # Clean JSON
            json_str = response.text
            if "```" in json_str:
                json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(json_str)
            
            st.toast("วิเคราะห์สำเร็จ!", icon="🎉")
            
            # --- แสดงผล ---
            t1, t2, t3, t4 = st.tabs(["📝 บทสรุป", "🕸️ ความสัมพันธ์", "📈 กราฟอารมณ์", "📊 ตัวละคร"])
            
            with t1:
                st.info(f"**เรื่องย่อ:** {data.get('summary')}")
                st.write(f"**แนวเรื่อง:** {data.get('genre')}")
                c1, c2 = st.columns(2)
                with c1: 
                    st.success("✅ **จุดแข็ง**")
                    for x in data.get('critique', {}).get('strengths', []): st.write(f"- {x}")
                with c2: 
                    st.error("❌ **จุดอ่อน**")
                    for x in data.get('critique', {}).get('weaknesses', []): st.write(f"- {x}")
                
                holes = data.get('critique', {}).get('plot_holes', [])
                if holes:
                    st.warning("**⚠️ ช่องโหว่ (Plot Holes):**")
                    for x in holes: st.write(f"- {x}")

            with t2:
                graph = graphviz.Digraph(attr={'rankdir':'LR'})
                for r in data.get('relations', []):
                    graph.edge(r.get('source','?'), r.get('target','?'), label=r.get('relation',''), penwidth=str(r.get('weight',1)/2))
                st.graphviz_chart(graph)

            with t3:
                df = pd.DataFrame(data.get('sentiment_arc', []))
                if not df.empty:
                    c = alt.Chart(df).mark_line(point=True).encode(x='chapter_part', y='score', tooltip=['mood'], color=alt.value('#FF4B4B')).interactive()
                    st.altair_chart(c, use_container_width=True)

            with t4:
                st.dataframe(pd.DataFrame(data.get('characters', [])), use_container_width=True)

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
            st.warning("ลองเปลี่ยนโมเดลอื่นในช่องเลือกด้านซ้ายดูนะครับ")
            if 'response' in locals():
                st.code(response.text)

else:
    if not novel_text:
        st.info("👈 1. อัปโหลดไฟล์นิยาย")
