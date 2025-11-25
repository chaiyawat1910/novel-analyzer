import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
import altair as alt
import graphviz
from pythainlp import word_tokenize
import io

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Super Novel Analyst (Gemini)", page_icon="✨", layout="wide")

st.title("✨ Super Novel Analyst: วิเคราะห์นิยายด้วย Gemini AI")
st.caption("ขับเคลื่อนด้วย Google Gemini - อ่านนิยายทั้งเรื่องในรวดเดียว!")

# --- Sidebar: ใส่กุญแจ API ---
with st.sidebar:
    st.header("🔑 ตั้งค่ากุญแจ (API Key)")
    api_key = st.text_input("วาง Google API Key ของคุณที่นี่:", type="password")
    st.markdown("[กดขอ API Key ฟรีที่นี่](https://aistudio.google.com/app/apikey)")
    st.info("💡 คำแนะนำ: หากเกิด Error ลองเช็คว่าก๊อปปี้ Key มาครบทุกตัวอักษรไหม")

# --- ส่วนรับข้อมูล ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. อัปโหลดนิยาย")
    uploaded_file = st.file_uploader("เลือกไฟล์นิยาย (.txt)", type=['txt'])
    novel_text = ""
    if uploaded_file:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        novel_text = stringio.read()
        st.success(f"✅ อ่านไฟล์สำเร็จ ({len(novel_text):,} ตัวอักษร)")

with col2:
    st.subheader("2. สั่งการ AI")
    if not api_key:
        st.warning("⚠️ กรุณาใสรหัส API Key ในช่องด้านซ้ายก่อนครับ")
        analyze_btn = False
    else:
        analyze_btn = st.button("🚀 ส่งให้ Gemini วิเคราะห์เดี๋ยวนี้!", type="primary", use_container_width=True)

# --- Logic การทำงาน ---
if analyze_btn and novel_text and api_key:
    
    # 1. ตั้งค่า Gemini
    genai.configure(api_key=api_key)
    
    # 2. เขียนคำสั่ง (Prompt)
    prompt = f"""
    คุณคือนักวิจารณ์วรรณกรรมและบรรณาธิการมืออาชีพ
    จงวิเคราะห์นิยายเรื่องนี้ (เนื้อหาอยู่ด้านล่าง) แล้วตอบกลับมาเป็น format JSON เท่านั้น ห้ามมีข้อความอื่นปน
    
    โครงสร้าง JSON ที่ต้องการ:
    {{
      "summary": "เขียนเรื่องย่อของนิยายเรื่องนี้แบบกระชับ ไม่เกิน 5 บรรทัด",
      "genre": "ระบุแนวของนิยาย",
      "characters": [
        {{"name": "ชื่อตัวละคร", "role": "บทบาท", "traits": "นิสัย"}}
      ],
      "relations": [
        {{"source": "ตัวละครA", "target": "ตัวละครB", "relation": "ความสัมพันธ์", "weight": 1-10}}
      ],
      "sentiment_arc": [
        {{"chapter_part": 1, "score": 10, "mood": "สดใส"}},
        {{"chapter_part": 2, "score": -5, "mood": "เครียด"}}
      ],
      "critique": {{
        "strengths": ["ข้อดี1", "ข้อดี2"],
        "weaknesses": ["ข้อเสีย1", "ข้อเสีย2"],
        "plot_holes": ["ช่องโหว่ (ถ้ามี)"]
      }}
    }}

    สำหรับ sentiment_arc ให้แบ่งเนื้อเรื่องเป็น 10 ส่วนเท่าๆ กัน แล้วประเมินคะแนน (-10 ถึง 10)

    --- เนื้อหานิยายเริ่มต้น ---
    {novel_text}
    --- เนื้อหานิยายสิ้นสุด ---
    """

    # 3. เริ่มส่งข้อมูล (พร้อมระบบกันแอพล่ม)
    with st.spinner('✨ Gemini กำลังอ่านนิยาย... (ระบบกำลังเลือกโมเดลที่ดีที่สุด)'):
        try:
            # พยายามใช้ 1.5 Flash (ตัวใหม่ เร็ว แม่น)
            model = genai.GenerativeModel('gemini-1.5-flash-latest', generation_config={"response_mime_type": "application/json"})
            response = model.generate_content(prompt)
        except Exception as e_flash:
            # ถ้าตัวใหม่พัง ให้ลองใช้ตัวเก่า (Pro) แทน
            try:
                st.warning(f"⚠️ โมเดล Flash มีปัญหา ({e_flash}) ...กำลังสลับไปใช้ Gemini Pro แทน")
                model = genai.GenerativeModel('gemini-pro') # ตัวนี้เสถียรสุดแต่อาจจะอ่านยาวมากไม่ได้
                response = model.generate_content(prompt)
                
                # Gemini Pro รุ่นเก่าอาจจะไม่ส่ง JSON เป๊ะๆ เราต้องแก้ข้อความนิดหน่อย
                if response.text.startswith("```json"):
                    json_str = response.text.strip("```json").strip("```")
                else:
                    json_str = response.text
            except Exception as e_pro:
                st.error(f"❌ เกิดข้อผิดพลาดร้ายแรง: {e_pro}")
                st.stop()

        # แปลงผลลัพธ์เป็น JSON
        try:
            # ตรวจสอบว่าใช้โมเดลตัวไหนส่งข้อมูลมา
            text_result = response.text
            # ล้าง format เผื่อ AI เผลอใส่ markdown มา
            text_result = text_result.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(text_result)
            
            # --- แสดงผลลัพธ์ ---
            st.toast("วิเคราะห์เสร็จสิ้น!", icon="🎉")
            
            # Tab 1: ภาพรวม
            t1, t2, t3, t4 = st.tabs(["📝 บทสรุป & วิจารณ์", "🕸️ ความสัมพันธ์", "📈 กราฟอารมณ์", "📊 ข้อมูลตัวละคร"])
            
            with t1:
                st.header(f"แนวเรื่อง: {data.get('genre', 'ไม่ระบุ')}")
                st.info(f"**เรื่องย่อ:** {data.get('summary')}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.success("✅ **จุดแข็ง**")
                    for item in data.get('critique', {}).get('strengths', []):
                        st.write(f"- {item}")
                with c2:
                    st.error("❌ **จุดอ่อน**")
                    for item in data.get('critique', {}).get('weaknesses', []):
                        st.write(f"- {item}")
                        
                if data.get('critique', {}).get('plot_holes'):
                    st.warning("**⚠️ ช่องโหว่ของพล็อต (Plot Holes):**")
                    for item in data['critique']['plot_holes']:
                        st.write(f"- {item}")

            with t2:
                st.header("แผนผังความสัมพันธ์")
                graph = graphviz.Digraph()
                graph.attr(rankdir='LR')
                
                for rel in data.get('relations', []):
                    graph.edge(rel.get('source', '?'), rel.get('target', '?'), 
                               label=rel.get('relation', ''), 
                               penwidth=str(float(rel.get('weight', 1))/2))
                    
                st.graphviz_chart(graph)

            with t3:
                st.header("เส้นทางอารมณ์")
                arc_data = pd.DataFrame(data.get('sentiment_arc', []))
                if not arc_data.empty:
                    chart = alt.Chart(arc_data).mark_line(point=True).encode(
                        x=alt.X('chapter_part', title='ช่วงเวลา (1-10)'),
                        y=alt.Y('score', title='คะแนนอารมณ์'),
                        tooltip=['chapter_part', 'mood', 'score'],
                        color=alt.value('#8A2BE2')
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.write("ไม่สามารถสร้างกราฟได้")

            with t4:
                st.header("ข้อมูลตัวละคร")
                chars = pd.DataFrame(data.get('characters', []))
                st.dataframe(chars, use_container_width=True)

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการแปลงข้อมูล: {e}")
            st.write("Raw Output:", response.text)

else:
    if not novel_text:
        st.info("👈 กรุณาอัปโหลดไฟล์นิยายทางซ้ายมือ")
