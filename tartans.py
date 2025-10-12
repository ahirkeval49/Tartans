# tartans.py
# CMU College of Engineering Program Navigator
# Author: Gemini (Google AI)
# Date: October 12, 2025
# Version: 1.5 (Implements critical URL logic and fixes bugs)

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
    """
    Scrapes a PREDEFINED LIST of CMU Engineering pages to find all graduate programs.
    Treats the main index page as a "critical" source for error reporting.
    """
    # --- BUG FIX: Renamed variable to source_urls and defined a separate base_url string ---
   critical_urls = 
        "https://engineering.cmu.edu/education/graduate-studies/programs/index.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/bme.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/cheme.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/cee.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/ece.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/epp.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/ini.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/iii.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/mse.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/meche.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/cmu-africa.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/sv.html"
    
    
    # --- NEW: Define the most important URL as critical ---
    critical_url = "https://engineering.cmu.edu/education/graduate-studies/programs/index.html"
    
    all_programs = {} # Use a dictionary to automatically handle duplicates by URL
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    progress_bar = st.progress(0, text="Initializing live data fetch from multiple CMU pages...")

    # --- BUG FIX: The loop variable must match the list name 'source_urls' ---
    for i, page_url in enumerate(source_urls):
        progress_bar.progress((i + 1) / len(source_urls), text=f"Scanning page: {page_url.split('/')[-1]}")
        try:
            response = requests.get(page_url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            program_elements = soup.select('div.component-content h4 a')
            
            for link in program_elements:
                program_name = link.text.strip()
                # --- BUG FIX: urljoin needs the base_url string, not the list ---
                program_url = urljoin(base_url, link['href'])
                
                if "department" in program_name.lower() or program_url == f"{base_url}/":
                    continue
                
                if program_url not in all_programs:
                     all_programs[program_url] = {'name': program_name, 'url': program_url, 'description': ''}

        except requests.RequestException as e:
            # --- NEW: Check if the failed URL is the critical one ---
            if page_url == critical_url:
                st.error(f"Critical source failed: Could not fetch main program page. Results may be incomplete. Error: {e}")
            else:
                st.warning(f"Could not fetch or process page {page_url.split('/')[-1]}: {e}")
            continue
    
    programs_list = list(all_programs.values())
    total_programs = len(programs_list)
    for i, program in enumerate(programs_list):
        progress_bar.progress((i + 1) / total_programs, text=f"Fetching details for: {program['name']}")
        try:
            sub_response = requests.get(program['url'], headers=headers, timeout=15)
            sub_soup = BeautifulSoup(sub_response.text, 'html.parser')
            description_tag = sub_soup.select_one('div.component-content p')
            program['description'] = description_tag.get_text(strip=True) if description_tag else 'No detailed description found.'
            time.sleep(0.1)
        except requests.RequestException:
            program['description'] = 'Could not retrieve description.'
            
    progress_bar.empty()
    if not programs_list:
        st.error("Scraping finished, but no program data was collected. All sources may be down or the website structure has changed.")
        return pd.DataFrame()
        
    st.success(f"Successfully scraped and processed {len(programs_list)} unique graduate programs.")
    return pd.DataFrame(programs_list)

# --- AI PROVIDER FUNCTIONS (using st.secrets for keys) ---

@st.cache_data
def get_deepseek_embedding(text: str) -> np.ndarray:
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not api_key: st.error("DeepSeek API Key is missing from secrets."); return None
    try:
        response = requests.post("https://api.deepseek.com/v1/embeddings", headers={"Authorization": f"Bearer {api_key}"}, json={"input": text, "model": "deepseek-embedding"}, timeout=20)
        response.raise_for_status(); return np.array(response.json()['data'][0]['embedding'])
    except Exception as e: st.error(f"DeepSeek Embedding API call failed: {e}"); return None

@st.cache_data
def get_gemini_embedding(text: str) -> np.ndarray:
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: st.error("Gemini API Key is missing from secrets."); return None
    try:
        genai.configure(api_key=api_key); result = genai.embed_content(model="models/embedding-001", content=text, task_type="RETRIEVAL_QUERY"); return np.array(result['embedding'])
    except Exception as e: st.error(f"Gemini Embedding API call failed: {e}"); return None

@st.cache_data
def get_deepseek_analysis(program_name: str, program_description: str, query: str) -> str:
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not api_key: return "AI analysis unavailable: DeepSeek API key missing."
    prompt = f"""As an expert academic advisor for Carnegie Mellon's College of Engineering, analyze the following graduate program for a prospective student interested in '{query}'.
    **Program:** {program_name} | **Description:** {program_description}
    Provide a concise, 3-point analysis in markdown format:
    - **Program Fit:** Explain why this program matches the student's interest in '{query}'.
    - **Key Application Skills:** What skills (e.g., Python, MATLAB, lab research) should the student highlight?
    - **Potential Career Trajectory:** Mention one or two specific job titles or industries."""
    try:
        response = requests.post("https://api.deepseek.com/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 400}, timeout=25)
        response.raise_for_status(); return response.json()['choices'][0]['message']['content']
    except Exception as e: return f"AI analysis failed to generate: {e}"

@st.cache_data
def get_gemini_analysis(program_name: str, program_description: str, query: str) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return "AI analysis unavailable: Gemini API key missing."
    prompt = f"""As an expert academic advisor for Carnegie Mellon's College of Engineering, analyze the following graduate program for a prospective student interested in '{query}'.
    **Program:** {program_name} | **Description:** {program_description}
    Provide a concise, 3-point analysis in markdown format:
    - **Program Fit:** Explain why this program matches the student's interest in '{query}'.
    - **Key Application Skills:** What skills (e.g., Python, MATLAB, lab research) should the student highlight?
    - **Potential Career Trajectory:** Mention one or two specific job titles or industries."""
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-pro-latest'); response = model.generate_content(prompt); return response.text
    except Exception as e: return f"AI analysis failed to generate: {e}"

# --- MAIN APPLICATION LOGIC ---
def main():
    st.title("🔧 CMU College of Engineering Program Navigator")
    st.markdown("Discover your ideal graduate program at Carnegie Mellon. Enter your interests below to get AI-powered recommendations based on live program data from the official CMU website.")

    with st.sidebar:
        st.image("https://www.cmu.edu/brand/brand-guidelines/assets/images/wordmarks-and-initials/cmu-wordmark-stacked-r-c.png", use_container_width=True)
        st.header("AI Configuration")
        
        available_providers = []
        if st.secrets.get("DEEPSEEK_API_KEY"): available_providers.append("DeepSeek")
        if st.secrets.get("GEMINI_API_KEY"): available_providers.append("Google Gemini")

        if not available_providers:
            st.error("No AI provider API key found in app secrets. The app administrator must add a key to enable AI features."); st.stop()

        if len(available_providers) > 1:
            ai_provider = st.selectbox("Choose AI Provider", available_providers, help="Select the AI model to power the recommendations.")
        else:
            ai_provider = available_providers[0]
            st.info(f"Using **{ai_provider}** API for analysis.")
        
        st.markdown("---")
        st.info("This tool is a proof-of-concept and not an official admissions tool.")

    embedding_function = get_deepseek_embedding if ai_provider == "DeepSeek" else get_gemini_embedding
    analysis_function = get_deepseek_analysis if ai_provider == "DeepSeek" else get_gemini_analysis

    df_programs = get_cmu_program_data()
    if df_programs.empty:
        st.warning("Program data could not be loaded. Please try again later."); return

    if 'embeddings_generated' not in st.session_state or st.session_state.get('ai_provider') != ai_provider:
        with st.spinner(f"🧠 Indexing programs using {ai_provider}..."):
            combined_text = df_programs.apply(lambda row: f"Program: {row['name']}\nDescription: {row['description']}", axis=1)
            df_programs['embedding'] = combined_text.apply(embedding_function)
            df_programs.dropna(subset=['embedding'], inplace=True)
            st.session_state.program_data = df_programs; st.session_state.embeddings_generated = True; st.session_state.ai_provider = ai_provider
    
    df = st.session_state.program_data
    
    search_query = st.text_input("**What are your academic and career interests?**", placeholder="e.g., 'robotics and automation', 'biomedical device design', 'machine learning for energy'")

    if search_query:
        with st.spinner(f"🔍 Analyzing your interests and finding the best matches with {ai_provider}..."):
            query_embedding = embedding_function(search_query)
            
            if query_embedding is not None:
                df['similarity'] = df['embedding'].apply(lambda x: np.dot(x, query_embedding) / (np.linalg.norm(x) * np.linalg.norm(query_embedding)))
                results = df.sort_values('similarity', ascending=False).head(3)

                st.subheader(f"Top 3 Program Matches for '{search_query}'")
                
                if results.empty:
                    st.warning("No strong matches found. Try rephrasing your interests for a better result.")
                else:
                    for i, (_, program) in enumerate(results.iterrows()):
                        st.markdown("---"); st.markdown(f"### **{i+1}. {program['name']}**")
                        col1, col2 = st.columns([3, 1])
                        with col1: st.progress(program['similarity'], text=f"**Match Score: {program['similarity']:.0%}**")
                        with col2: st.link_button("Go to Program Website ↗️", program['url'], use_container_width=True)
                        with st.expander("**Program Overview from CMU Website**"): st.write(program['description'])
                        with st.spinner("🤖 Generating AI Advisor analysis..."):
                            analysis = analysis_function(program['name'], program['description'], search_query)
                        st.markdown("**AI-Powered Advisor Analysis**"); st.info(analysis)
            else:
                st.error("Could not process your query. The AI embedding could not be generated.")

if __name__ == "__main__":
    main()
