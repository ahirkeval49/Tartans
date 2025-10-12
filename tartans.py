# tartans.py
# CMU College of Engineering Program Navigator
# Author: Gemini (Google AI)
# Date: October 12, 2025
# Version: 2.2 (Implements a resilient, content-based scraper)

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
    page_icon="🎓",
    layout="wide"
)

# --- DATA SCRAPING & CACHING ---
@st.cache_data(ttl=86400) # Cache data for 24 hours
def get_cmu_program_data():
    """
    Scrapes a PREDEFINED LIST of CMU Engineering pages.
    NEW RESILIENT STRATEGY: Selects all links in the main content area and filters them
    based on keywords, which is much more robust than relying on fragile HTML structure.
    """
    base_url = "https://engineering.cmu.edu"
    source_urls = [
        "https://engineering.cmu.edu/education/graduate-studies/programs/index",
        "https://engineering.cmu.edu/education/graduate-studies/programs/bme",
        "https://engineering.cmu.edu/education/graduate-studies/programs/cheme",
        "https://engineering.cmu.edu/education/graduate-studies/programs/cee",
        "https://engineering.cmu.edu/education/graduate-studies/programs/ece",
        "https://engineering.cmu.edu/education/graduate-studies/programs/epp",
        "https://engineering.cmu.edu/education/graduate-studies/programs/ini",
        "https://engineering.cmu.edu/education/graduate-studies/programs/iii",
        "https://engineering.cmu.edu/education/graduate-studies/programs/mse",
        "https://engineering.cmu.edu/education/graduate-studies/programs/meche",
        "https://engineering.cmu.edu/education/graduate-studies/programs/cmu-africa",
        "https://engineering.cmu.edu/education/graduate-studies/programs/sv"
    ]
    critical_url = "https://engineering.cmu.edu/education/graduate-studies/programs/index.html"
    all_programs = {}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    progress_bar = st.progress(0, text="Initializing live data fetch...")
    
    for i, page_url in enumerate(source_urls):
        progress_bar.progress((i + 1) / len(source_urls), text=f"Scanning page: {page_url.split('/')[-1]}")
        try:
            response = requests.get(page_url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # --- RESILIENT SELECTOR ---
            # 1. Target the main content container by its unique ID.
            content_area = soup.find('div', id='content_a')
            if not content_area:
                continue # Skip this page if the main content div isn't found
            
            # 2. Find ALL links within that container.
            all_links = content_area.find_all('a', href=True)

            for link in all_links:
                program_name = link.text.strip()
                href = link['href']
                
                # 3. Intelligently filter the links to find actual programs.
                is_program_link = ('M.S.' in program_name or 'Ph.D.' in program_name or 'Master' in program_name)
                is_valid_path = href.startswith('/education/graduate-studies/programs/')
                
                if is_program_link and is_valid_path:
                    program_url = urljoin(base_url, href)
                    degree_type = 'Other'
                    if 'M.S.' in program_name or 'Master' in program_name: degree_type = 'M.S.'
                    elif 'Ph.D.' in program_name: degree_type = 'Ph.D.'
                    
                    if program_url not in all_programs:
                        all_programs[program_url] = {'name': program_name, 'url': program_url, 'description': '', 'degree_type': degree_type}
        except requests.RequestException as e:
            if page_url == critical_url: st.error(f"Critical source failed: Could not fetch main program page. Error: {e}")
            else: st.warning(f"Could not fetch or process page {page_url.split('/')[-1]}: {e}")
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

# --- AI PROVIDER FUNCTIONS ---

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
def get_deepseek_analysis(program_name: str, program_description: str, student_profile: str) -> str:
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not api_key: return "AI analysis unavailable: DeepSeek API key missing."
    prompt = f"""As an expert CMU academic advisor, analyze this program for a student with the following profile:
    **Student Profile:** '{student_profile}'
    
    **Program to Analyze:** {program_name}
    **Description:** {program_description}

    Provide a concise, 3-point analysis in markdown format, tailored to the student's profile:
    - **Program Fit:** How well does this program align with the student's stated background and goals?
    - **Key Application Skills:** Based on their profile, what specific skills should this student highlight in their application?
    - **Potential Career Trajectory:** How does this degree help them achieve their specific career ambitions?"""
    try:
        response = requests.post("https://api.deepseek.com/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}"}, json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 400}, timeout=25)
        response.raise_for_status(); return response.json()['choices'][0]['message']['content']
    except Exception as e: return f"AI analysis failed to generate: {e}"

@st.cache_data
def get_gemini_analysis(program_name: str, program_description: str, student_profile: str) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return "AI analysis unavailable: Gemini API key missing."
    prompt = f"""As an expert CMU academic advisor, analyze this program for a student with the following profile:
    **Student Profile:** '{student_profile}'
    
    **Program to Analyze:** {program_name}
    **Description:** {program_description}

    Provide a concise, 3-point analysis in markdown format, tailored to the student's profile:
    - **Program Fit:** How well does this program align with the student's stated background and goals?
    - **Key Application Skills:** Based on their profile, what specific skills should this student highlight in their application?
    - **Potential Career Trajectory:** How does this degree help them achieve their specific career ambitions?"""
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-1.5-pro-latest'); response = model.generate_content(prompt); return response.text
    except Exception as e: return f"AI analysis failed to generate: {e}"

# --- MAIN APPLICATION LOGIC ---
def main():
    st.title("🎓 CMU Engineering Program Navigator")
    st.markdown("Answer a few questions to discover the graduate program that best fits your academic and career goals.")

    with st.sidebar:
        st.image("https://www.cmu.edu/brand/brand-guidelines/assets/images/wordmarks-and-initials/cmu-wordmark-stacked-r-c.png", use_container_width=True)
        st.header("AI Configuration")
        available_providers = []
        if st.secrets.get("DEEPSEEK_API_KEY"): available_providers.append("DeepSeek")
        if st.secrets.get("GEMINI_API_KEY"): available_providers.append("Google Gemini")
        if not available_providers: st.error("No AI provider API key found in app secrets."); st.stop()
        ai_provider = st.selectbox("Choose AI Provider", available_providers) if len(available_providers) > 1 else available_providers[0]
        st.info(f"Using **{ai_provider}** API for analysis.")
        st.markdown("---")
        st.info("This tool is a proof-of-concept and not an official admissions tool.")

    embedding_function = get_deepseek_embedding if ai_provider == "DeepSeek" else get_gemini_embedding
    analysis_function = get_deepseek_analysis if ai_provider == "DeepSeek" else get_gemini_analysis

    df_programs = get_cmu_program_data()
    if df_programs.empty:
        st.warning("Program data could not be loaded. Please try again later."); return

    # --- Interactive Questionnaire ---
    st.subheader("Tell us about yourself:")
    
    common_majors = ["Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Chemical Engineering", "Biomedical Engineering", "Materials Science", "Physics", "Mathematics", "Other"]
    
    with st.form("student_profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            degree_level = st.radio("What degree level are you pursuing?", ("M.S.", "Ph.D."), horizontal=True)
            background = st.multiselect("What is your academic background? (Select all that apply)", options=common_majors, default=["Mechanical Engineering"])
        with col2:
            career_goal = st.selectbox("What is your primary career ambition?", 
                                       ("Industry Leadership (e.g., Tech, Manufacturing, Energy)", 
                                        "Research & Academia (e.g., Professor, National Lab Scientist)",
                                        "Startup & Entrepreneurship",
                                        "Government & Public Policy"))
        
        learning_style = st.slider(
            "What's your preferred learning style?",
            0, 100, 50,
            help="0 = Purely Theoretical/Research, 100 = Highly Applied/Project-Based"
        )
        
        keywords = st.text_area(
            "List specific keywords, topics, or technologies that interest you.",
            placeholder="e.g., machine learning, robotics, sustainable energy, quantum computing, battery technology, medical devices..."
        )
        
        submitted = st.form_submit_button("🎓 Find My Program", use_container_width=True)

    if submitted:
        style_desc = ""
        if learning_style < 20: style_desc = "a heavily theoretical and research-focused program"
        elif learning_style < 40: style_desc = "a program with a strong theoretical basis"
        elif learning_style < 60: style_desc = "a balanced program with both theory and hands-on projects"
        elif learning_style < 80: style_desc = "a project-driven program with a solid theoretical foundation"
        else: style_desc = "a highly applied, hands-on, and project-based program"
        
        synthesized_query = (
            f"I am a student with a background in {', '.join(background)} looking for a {degree_level} program. "
            f"My primary career goal is {career_goal.split('(')[0].strip()}. "
            f"I am most interested in topics like {keywords}. "
            f"I thrive in {style_desc}."
        )

        with st.expander("Your Generated Profile for AI Matching"):
            st.write(synthesized_query)
        
        if 'embeddings_generated' not in st.session_state or st.session_state.get('ai_provider') != ai_provider:
            with st.spinner(f"🧠 Indexing programs using {ai_provider}..."):
                df_programs['embedding'] = df_programs.apply(lambda row: embedding_function(f"Program: {row['name']}\nDescription: {row['description']}"), axis=1)
                df_programs.dropna(subset=['embedding'], inplace=True)
                st.session_state.program_data = df_programs; st.session_state.embeddings_generated = True; st.session_state.ai_provider = ai_provider
        
        df = st.session_state.program_data
        
        with st.spinner(f"🔍 Analyzing your profile and finding the best matches with {ai_provider}..."):
            query_embedding = embedding_function(synthesized_query)
            if query_embedding is not None:
                df['similarity'] = df['embedding'].apply(lambda x: np.dot(x, query_embedding) / (np.linalg.norm(x) * np.linalg.norm(query_embedding)))
                df['degree_match_boost'] = df.apply(lambda row: 0.1 if row['degree_type'] == degree_level else 0, axis=1)
                df['final_score'] = df['similarity'] + df['degree_match_boost']
                
                results = df.sort_values('final_score', ascending=False).head(3)

                st.subheader(f"Top 3 Program Recommendations")
                if results.empty:
                    st.warning("No strong matches found. Try adjusting your criteria.")
                else:
                    for i, (_, program) in enumerate(results.iterrows()):
                        st.markdown("---"); st.markdown(f"### **{i+1}. {program['name']}**")
                        col1, col2 = st.columns([3, 1])
                        with col1: st.progress(program['similarity'], text=f"**Profile Match Score: {program['similarity']:.0%}**")
                        with col2: st.link_button("Go to Program Website ↗️", program['url'], use_container_width=True)
                        with st.expander("**Program Overview from CMU Website**"): st.write(program['description'])
                        with st.spinner("🤖 Generating personalized AI Advisor analysis..."):
                            analysis = analysis_function(program['name'], program['description'], synthesized_query)
                        st.markdown("**AI-Powered Advisor Analysis**"); st.info(analysis)
            else:
                st.error("Could not process your query.")

if __name__ == "__main__":
    main()
