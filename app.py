import streamlit as st
import os
import shutil
import time
from graph import app
from ingest import ingest_data
import json
import re
import pandas as pd

# --- PERSISTENT MEMORY ---
LOG_FILE = "doc_log.txt"

def get_uploaded_files():
    if not os.path.exists(LOG_FILE): return []
    with open(LOG_FILE, "r") as f: return [line.strip() for line in f.readlines()]

def add_file_to_log(filename):
    current_files = get_uploaded_files()
    if filename not in current_files:
        with open(LOG_FILE, "a") as f: f.write(filename + "\n")

def clear_log():
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Artha | Financial Intelligence", page_icon="📈", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Jaini&display=swap');
    .stApp { background-color: #FFFFFF; color: #000000; }
    .artha-logo { font-family: 'Jaini', system-ui; font-size: 80px; color: #FF9F1C; text-align: right; line-height: 0.8; }
    .tagline { font-family: 'Helvetica Neue', sans-serif; font-size: 16px; color: #555; text-align: right; letter-spacing: 1px; }
    .greeting-text { font-family: 'Helvetica Neue', sans-serif; font-size: 50px; font-weight: 700; color: #333; margin-top: 60px; }
    .sub-greeting { font-family: 'Helvetica Neue', sans-serif; font-size: 18px; color: #666; margin-bottom: 40px; }
    [data-testid="stSidebar"] { background-color: #F8F9FA; border-right: 1px solid #E0E0E0; }
    .stTextInput input { color: #000; background-color: #FFF; border: 1px solid #CCC; }
    header {visibility: hidden;} .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.subheader("📂 Document Library")
    existing_files = get_uploaded_files()
    if not existing_files:
        st.info("Library Empty. Upload a PDF.")
        selected_file = "All Documents"
    else:
        options = ["All Documents"] + existing_files
        selected_file = st.selectbox("Active Context:", options)

    st.markdown("---")
    with st.expander("➕ Add New Report", expanded=True):
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
        if uploaded_file:
            file_name = uploaded_file.name
            if not os.path.exists("./temp_pdf"): os.makedirs("./temp_pdf")
            save_path = os.path.join("./temp_pdf", file_name)
            with open(save_path, "wb") as f: f.write(uploaded_file.getbuffer())
            
            if st.button("Ingest Data", use_container_width=True):
                with st.spinner("Processing..."):
                    try:
                        ingest_data(save_path, file_name)
                        add_file_to_log(file_name)
                        st.success("Done!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

    st.markdown("---")
    if st.button("🗑️ Clear Memory", use_container_width=True):
        if os.path.exists("./chroma_db"): shutil.rmtree("./chroma_db")
        clear_log()
        st.rerun()

# --- MAIN LAYOUT ---
c1, c2 = st.columns([7, 3]) 
with c2:
    st.markdown('<div class="artha-logo">अर्थ</div><div class="tagline">Financial Intelligence</div>', unsafe_allow_html=True)

st.markdown('<div class="greeting-text">Analyze any company,<br>instantly.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-greeting">Upload a 10-K report or ask about live market trends.</div>', unsafe_allow_html=True)

# --- CHAT ---
if "messages" not in st.session_state: st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt := st.chat_input("Ask about Revenue, Risks, or Live Stock Prices..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("⏳ *Thinking...*")
        full_response = ""
        source_used = "Unknown"

        try:
            inputs = {"question": prompt, "file_filter": selected_file}
            for output in app.stream(inputs):
                for key, value in output.items():
                    # --- Status Updates ---
                    if key == "decompose_query":
                        response_placeholder.markdown("🧠 *Decomposing Question...*")
                    elif key == "retrieve": 
                        response_placeholder.markdown(f"📂 *Reading {selected_file}...*")
                    elif key == "grade_documents":
                        if value.get("web_search") == "Yes": response_placeholder.markdown("🌍 *Searching Live Web...*")
                        else: response_placeholder.markdown("✅ *Found data in PDF...*")
                    elif key == "web_search_node":
                        response_placeholder.markdown("🌍 *Scanning Internet...*")
                        source_used = "Live Web Search"
                    
                    # --- Final Answer Generation ---
                    elif key == "generate":
                        full_response = value["generation"]
                        
                        # --- CHART DETECTION LOGIC ---
                        chart_match = re.search(r"```json\s*({.*?})\s*```", full_response, re.DOTALL)
                        
                        if chart_match:
                            try:
                                # 1. Extract and Parse JSON
                                json_str = chart_match.group(1)
                                chart_data = json.loads(json_str)
                                
                                # 2. Clean up text (remove the raw JSON so user doesn't see it)
                                clean_text = full_response.replace(chart_match.group(0), "")
                                response_placeholder.markdown(clean_text)
                                
                                # 3. Render the Chart
                                if "bar_chart" in chart_data:
                                    data = chart_data["bar_chart"]
                                    # Create DataFrame for Streamlit
                                    df = pd.DataFrame({
                                        "Entity": data["labels"],
                                        data["datasets"][0]["label"]: data["datasets"][0]["data"]
                                    }).set_index("Entity")
                                    
                                    # Render Bar Chart
                                    st.bar_chart(df)
                                    st.caption("📊 Visualized by Artha AI")
                                    
                            except Exception as e:
                                # Fallback: If chart fails, just show text
                                response_placeholder.markdown(full_response)
                        else:
                            # No chart found, just show text
                            response_placeholder.markdown(full_response)

            # Save the CLEANED text (without JSON) to chat history
            # If a chart was present, we stripped it above. If not, full_response is just text.
            # (Simplification: We save the full response to history for now)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"An error occurred: {e}")