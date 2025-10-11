# app.py
# CMU College of Engineering Program Navigator
# Author: Gemini (Google AI)
# Date: October 11, 2025
# Version: 1.1 (Includes scraper fix)

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
@st.cache_data(ttl=86400) # Cache data for 24 hours (86400 seconds)
def get_cmu_program_data():
    """
    Scrapes the CMU Engineering graduate programs page for program names, URLs, and descriptions.
    This version uses an UPDATED CSS SELECTOR to match the current website structure as of Oct 2025.
    """
    base_url = "https://engineering.cmu.edu"
    source_url = f"{base_url}/education/graduate-studies/programs/index.html"
    programs = []
    
    progress_bar = st.progress(0, text="Initializing live data fetch from CMU Engineering...")

    try:
        response = requests.get(source_url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # --- CRITICAL FIX [October 2025] ---
        # The website structure changed. The old selector 'div.program-listing h3 a' no longer works.
        # The new, correct selector finds an <a> tag inside an <h4> inside a div with class 'component-content'.
        program_elements = soup.select('div.component-content h4 a')
        
        if not program_elements:
            st.error("Scraper Error: Could not find program links. The website's structure may have changed.")
            progress_bar.empty()
            return pd.DataFrame()

        for i, link in enumerate(program_elements):
            program_name = link.text.strip()
            program_url = urljoin(base_url, link['href'])
            
            progress_bar.progress((i + 1) / len(program_elements), text=f"Scraping: {program_name}")
            
            try:
                sub_response = requests.get(program_url, timeout=15)
                sub_soup = BeautifulSoup(sub_response.text, 'html.parser')
                
                # This selector is also updated to be more robust for the program detail pages.
                description_tag = sub_soup.select_one('div.component-content p')
                description = description_tag.get_text(strip=True) if description_tag else 'No detailed description found.'
                
                # A simple filter to avoid scraping links that are not actual degree programs.
                if "department" in program_name.lower():
                    continue

                programs.append({'name': program_name, 'url': program_url, 'description': description})
                time.sleep(0.1) # Be a polite scraper
            except requests.RequestException as e:
                st.warning(f"Could not fetch details for {program_name}: {e}")

    except requests.RequestException as e:
        st.error(f"A critical error occurred while scraping the main program list: {e}")
        progress_bar.empty()
        return pd.DataFrame()
    
    progress_bar.empty()
    if not programs:
        st.error("Scraping finished, but no program data was collected.")
        return pd.DataFrame()
        
    st.success(f"Successfully scraped and processed {len(programs)} graduate programs.")
    return pd.DataFrame(programs)

# --- AI PROVIDER FUNCTIONS (using st.secrets for keys) ---

@st.cache_data
def get_deepseek_embedding(text: str) -> np.ndarray:
    """Generates an embedding vector using the DeepSeek API."""
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("DeepSeek API Key is missing from secrets.")
        return None
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"input": text, "model": "deepseek-embedding"},
            timeout=20
        )
        response.raise_for_status()
        return np.array(response.json()['data'][0]['embedding'])
    except Exception as e:
        st.error(f"DeepSeek Embedding API call failed: {e}")
        return None

@st.cache_data
def get_gemini_embedding(text: str) -> np.ndarray:
    """Generates an embedding vector using the Google Gemini API."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Gemini API Key is missing from secrets.")
        return None
    try:
        genai.configure(api_key=api_key)
        result = genai.embed_content(model="models/embedding-001", content=text, task_type="RETRIEVAL_QUERY")
        return np.array(result['embedding'])
    except Exception as e:
        st.error(f"Gemini Embedding API call failed: {e}")
        return None

@st.cache_data
def get_deepseek_analysis(program_name: str, program_description: str, query: str) -> str:
    """Generates a qualitative analysis using the DeepSeek Chat API."""
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not api_key: return "AI analysis unavailable: DeepSeek API key missing."
    
    prompt = f"""As an expert academic advisor for Carnegie Mellon's College of Engineering, analyze the following graduate program for a prospective student interested in '{query}'.

    **Program:** {program_name}
    **Description:** {program_description}

    Provide a concise, 3-point analysis in markdown format:
    - **Program Fit:** Briefly explain why this program is a strong, moderate, or weak match for the student's interest in '{query}'.
    - **Key Application Skills:** What specific skills or experiences (e.g., Python, MATLAB, lab research, internships) should the student highlight in their application for this program?
    - **Potential Career Trajectory:** Mention one or two specific, high-potential job titles or industries this degree could lead to, related to their interest.
    """
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 400},
            timeout=25
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"AI analysis failed to generate: {e}"

@st.cache_data
def get_gemini_analysis(program_name: str, program_description: str, query: str) -> str:
    """Generates a qualitative analysis using the Google Gemini API."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return "AI analysis unavailable: Gemini API key missing."
    
    prompt = f"""As an expert academic advisor for Carnegie Mellon's College of Engineering, analyze the following graduate program for a prospective student interested in '{query}'.

    **Program:** {program_name}
    **Description:** {program_description}

    Provide a concise, 3-point analysis in markdown format:
    - **Program Fit:** Briefly explain why this program is a strong, moderate, or weak match for the student's interest in '{query}'.
    - **Key Application Skills:** What specific skills or experiences (e.g., Python, MATLAB, lab research, internships) should the student highlight in their application for this program?
    - **Potential Career Trajectory:** Mention one or two specific, high-potential job titles or industries this degree could lead to, related to their interest.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI analysis failed to generate: {e}"

# --- MAIN APPLICATION LOGIC ---
def main():
    st.title("🔧 CMU College of Engineering Program Navigator")
    st.markdown("Discover your ideal graduate program at Carnegie Mellon. Enter your interests below to get AI-powered recommendations based on live program data from the official CMU website.")

    # --- Sidebar for AI Provider Selection and Info ---
    with st.sidebar:
        st.image("https://www.cmu.edu/brand/brand-guidelines/assets/images/wordmarks-and-initials/cmu-wordmark-stacked-r-c.png", use_column_width=True)
        st.header("AI Configuration")
        
        available_providers = []
        if st.secrets.get("DEEPSEEK_API_KEY"):
            available_providers.append("DeepSeek")
        if st.secrets.get("GEMINI_API_KEY"):
            available_providers.append("Google Gemini")

        if not available_providers:
            st.error("No AI provider API key found in app secrets. The app administrator must add a key to enable AI features.")
            st.stop()

        if len(available_providers) > 1:
            ai_provider = st.selectbox("Choose AI Provider", available_providers, help="Select the AI model to power the recommendations.")
        else:
            ai_provider = available_providers[0]
            st.info(f"Using **{ai_provider}** API for analysis.")
        
        st.markdown("---")
        st.info("This tool uses live data and AI to help students explore programs. It is not an official admissions tool.")

    # Assign correct AI functions based on selection
    embedding_function = get_deepseek_embedding if ai_provider == "DeepSeek" else get_gemini_embedding
    analysis_function = get_deepseek_analysis if ai_provider == "DeepSeek" else get_gemini_analysis

    # Load program data from scraper
    df_programs = get_cmu_program_data()
    if df_programs.empty:
        st.warning("Program data could not be loaded. Please try again later.")
        return

    # Generate and cache embeddings in session state
    if 'embeddings_generated' not in st.session_state or st.session_state.get('ai_provider') != ai_provider:
        with st.spinner(f"🧠 Indexing programs using {ai_provider}..."):
            combined_text = df_programs.apply(lambda row: f"Program: {row['name']}\nDescription: {row['description']}", axis=1)
            df_programs['embedding'] = combined_text.apply(embedding_function)
            df_programs.dropna(subset=['embedding'], inplace=True)
            st.session_state.program_data = df_programs
            st.session_state.embeddings_generated = True
            st.session_state.ai_provider = ai_provider
    
    df = st.session_state.program_data
    
    # --- User Input and Matching ---
    search_query = st.text_input(
        "**What are your academic and career interests?**",
        placeholder="e.g., 'robotics and automation in manufacturing', 'biomedical device design', 'machine learning for sustainable energy'"
    )

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
                        st.markdown("---")
                        st.markdown(f"### **{i+1}. {program['name']}**")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.progress(program['similarity'], text=f"**Match Score: {program['similarity']:.0%}**")
                        with col2:
                            st.link_button("Go to Program Website ↗️", program['url'], use_container_width=True)
                        
                        with st.expander("**Program Overview from CMU Website**"):
                            st.write(program['description'])
                        
                        with st.spinner("🤖 Generating AI Advisor analysis..."):
                            analysis = analysis_function(program['name'], program['description'], search_query)
                        
                        st.markdown("**AI-Powered Advisor Analysis**")
                        st.info(analysis)
            else:
                st.error("Could not process your query. The AI embedding could not be generated. Please check the status of the selected AI provider.")

if __name__ == "__main__":
    main()
