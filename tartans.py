# app.py

import streamlit as st
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CMU Engineering Program Navigator",
    page_icon="🔧",
    layout="wide"
)

# --- DATA SCRAPING & CACHING ---
@st.cache_data(ttl=86400) # Cache data for 24 hours
def get_cmu_program_data():
    """Scrapes the CMU Engineering graduate programs page for program details."""
    base_url = "https://engineering.cmu.edu"
    source_url = f"{base_url}/education/graduate-studies/programs/index.html"
    programs = []
    
    progress_text = "Fetching live data from CMU Engineering... This may take a moment on the first run."
    progress_bar = st.progress(0, text=progress_text)

    try:
        response = requests.get(source_url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        program_elements = soup.select('div.program-listing h3 a') # Using CSS selector
        
        if not program_elements:
            st.error("Scraper Error: Could not find program links. The website's structure may have changed.")
            return pd.DataFrame()

        for i, link in enumerate(program_elements):
            program_name = link.text.strip()
            # Use urljoin to handle relative links correctly
            program_url = urljoin(base_url, link['href'])
            
            try:
                sub_response = requests.get(program_url, timeout=15)
                sub_soup = BeautifulSoup(sub_response.text, 'html.parser')
                
                # Find the introductory paragraph(s). This selector is more robust.
                description_tag = sub_soup.select_one('div.text.parbase.section > p')
                description = description_tag.get_text(strip=True) if description_tag else 'No detailed description found.'
                
                programs.append({
                    'name': program_name,
                    'url': program_url,
                    'description': description
                })
                time.sleep(0.1) # Be polite to the server
            except requests.RequestException as e:
                # If a sub-page fails, we can skip it and continue
                st.warning(f"Could not fetch details for {program_name}: {e}")

            progress_bar.progress((i + 1) / len(program_elements), text=f"Scraping: {program_name}")

    except requests.RequestException as e:
        st.error(f"A critical error occurred during scraping the main page: {e}")
        return pd.DataFrame()
    
    progress_bar.empty() # Clear the progress bar
    if not programs:
        st.error("Scraping finished, but no program data was collected.")
        return pd.DataFrame()
        
    st.success(f"Successfully scraped and processed {len(programs)} programs.")
    return pd.DataFrame(programs)

# --- AI EMBEDDING FUNCTIONS ---
@st.cache_data
def get_deepseek_embedding(text: str, api_key: str) -> np.ndarray:
    # ... (Paste the DeepSeek embedding function from above) ...
    DEEPSEEK_EMBEDDING_URL = "https://api.deepseek.com/v1/embeddings"
    if not api_key: st.error("DeepSeek API Key missing."); return None
    try:
        response = requests.post(DEEPSEEK_EMBEDDING_URL, headers={"Authorization": f"Bearer {api_key}"}, json={"input": text, "model": "deepseek-embedding"}, timeout=20)
        response.raise_for_status(); return np.array(response.json()['data'][0]['embedding'])
    except Exception as e: st.error(f"DeepSeek Embedding API call failed: {e}"); return None

@st.cache_data
def get_gemini_embedding(text: str, api_key: str) -> np.ndarray:
    # ... (Paste the Gemini embedding function from above) ...
    if not api_key: st.error("Gemini API Key missing."); return None
    try:
        genai.configure(api_key=api_key)
        result = genai.embed_content(model="models/embedding-001", content=text, task_type="RETRIEVAL_QUERY")
        return np.array(result['embedding'])
    except Exception as e: st.error(f"Gemini Embedding API call failed: {e}"); return None

# --- AI ANALYSIS FUNCTIONS ---
@st.cache_data
def get_deepseek_analysis(program_name: str, program_description: str, query: str, api_key: str) -> str:
    # ... (Paste the DeepSeek analysis function from above) ...
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
    if not api_key: return "AI analysis unavailable: API key missing."
    prompt = f"As an expert CMU academic advisor, analyze the following graduate program for a prospective student interested in '{query}'.\n\n**Program:** {program_name}\n**Description:** {program_description}\n\nProvide a concise, 3-point analysis in markdown format:\n- **Program Fit:** Briefly explain why this program is a strong, moderate, or weak match for the student's interest in '{query}'.\n- **Key Application Skills:** What specific skills or experiences should the student highlight in their application?\n- **Potential Career Trajectory:** Mention one or two specific, high-potential job titles or industries this degree could lead to."
    try:
        response = requests.post(DEEPSEEK_API_URL, headers={"Authorization": f"Bearer {api_key}"}, json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 400}, timeout=25)
        response.raise_for_status(); return response.json()['choices'][0]['message']['content']
    except Exception as e: return f"AI analysis failed: {e}"

@st.cache_data
def get_gemini_analysis(program_name: str, program_description: str, query: str, api_key: str) -> str:
    # ... (Paste the Gemini analysis function from above) ...
    if not api_key: return "AI analysis unavailable: API key missing."
    genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"As an expert CMU academic advisor, analyze the following graduate program for a prospective student interested in '{query}'.\n\n**Program:** {program_name}\n**Description:** {program_description}\n\nProvide a concise, 3-point analysis in markdown format:\n- **Program Fit:** Briefly explain why this program is a strong, moderate, or weak match for the student's interest in '{query}'.\n- **Key Application Skills:** What specific skills or experiences should the student highlight in their application?\n- **Potential Career Trajectory:** Mention one or two specific, high-potential job titles or industries this degree could lead to."
    try:
        response = model.generate_content(prompt); return response.text
    except Exception as e: return f"AI analysis failed: {e}"

# --- MAIN APP LOGIC ---
def main():
    st.title("🔧 CMU College of Engineering Program Navigator")
    st.markdown("Discover your ideal graduate program at Carnegie Mellon. Enter your interests below to get AI-powered recommendations based on live program data.")

    # --- Sidebar for configuration ---
    with st.sidebar:
        st.header("Configuration")
        ai_provider = st.selectbox("Choose your AI Provider", ["DeepSeek", "Google Gemini"])
        
        api_key = ""
        if ai_provider == "DeepSeek":
            api_key = st.text_input("Enter your DeepSeek API Key", type="password", help="Get yours from https://platform.deepseek.com/")
        elif ai_provider == "Google Gemini":
            api_key = st.text_input("Enter your Google Gemini API Key", type="password", help="Get yours from https://aistudio.google.com/app/apikey")
        
        st.markdown("---")
        st.info("Your API key is used only for this session and is not stored.")

    # --- Load and process data ---
    df_programs = get_cmu_program_data()

    if df_programs.empty:
        st.warning("Program data could not be loaded. The tool cannot proceed.")
        return

    if not api_key:
        st.warning(f"Please enter your {ai_provider} API key in the sidebar to begin.")
        return

    # Select the correct functions based on user choice
    embedding_function = get_deepseek_embedding if ai_provider == "DeepSeek" else get_gemini_embedding
    analysis_function = get_deepseek_analysis if ai_provider == "DeepSeek" else get_gemini_analysis

    # Generate embeddings once per session
    if 'embeddings_generated' not in st.session_state or st.session_state.get('ai_provider') != ai_provider:
        with st.spinner(f"🧠 Generating program embeddings using {ai_provider}..."):
            df_programs['embedding'] = df_programs.apply(
                lambda row: embedding_function(f"Program: {row['name']}\nDescription: {row['description']}", api_key),
                axis=1
            )
            df_programs.dropna(subset=['embedding'], inplace=True)
            st.session_state.program_data = df_programs
            st.session_state.embeddings_generated = True
            st.session_state.ai_provider = ai_provider
    
    df = st.session_state.program_data
    
    # --- User Input and Matching ---
    search_query = st.text_input(
        "**What are your academic and career interests?**",
        placeholder="e.g., 'sustainable energy systems', 'machine learning in healthcare', 'robotics for space exploration'"
    )

    if search_query:
        with st.spinner(f"🔍 Analyzing your interests and finding the best matches..."):
            query_embedding = embedding_function(search_query, api_key)
            
            if query_embedding is not None:
                # Cosine similarity calculation
                df['similarity'] = df['embedding'].apply(lambda x: np.dot(x, query_embedding) / (np.linalg.norm(x) * np.linalg.norm(query_embedding)))
                
                results = df.sort_values('similarity', ascending=False).head(3)

                st.subheader(f"Top 3 Program Matches for '{search_query}'")
                
                if results.empty:
                    st.warning("No strong matches found. Try rephrasing your interests.")
                else:
                    for i, (_, program) in enumerate(results.iterrows()):
                        st.markdown(f"---")
                        st.markdown(f"### **{i+1}. {program['name']}**")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.progress(program['similarity'], text=f"**Match Score: {program['similarity']:.0%}**")
                        with col2:
                            st.link_button("Go to Program Website ↗️", program['url'], use_container_width=True)

                        with st.expander("**Program Overview**"):
                            st.write(program['description'])
                        
                        with st.spinner("🤖 Generating AI Advisor analysis..."):
                            analysis = analysis_function(program['name'], program['description'], search_query, api_key)
                        
                        st.markdown(analysis)
            else:
                st.error("Could not process your query. Please check your API key and try again.")

if __name__ == "__main__":
    main()
