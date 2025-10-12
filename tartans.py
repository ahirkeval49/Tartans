# tartans.py
# CMU College of Engineering Program Navigator
# Author: Gemini (Google AI)
# Date: October 13, 2025 (Updated for CMU 2025 Site Structure)
# Version: 2.4 (Improved Scraping Resilience)

import streamlit as st
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin
import google.generativeai as genai
from typing import Optional, Dict, Any, List

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CMU Engineering Program Navigator",
    page_icon="🎓",
    layout="wide"
)

# --- UTILITIES ---

def robust_request(url: str, headers: Dict[str, str], timeout: int = 15) -> Optional[str]:
    """Handles HTTP requests with basic error checking."""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        # print(f"Request failed for {url}: {e}")
        return None

# --- DATA SCRAPING & CACHING ---
@st.cache_data(ttl=86400) # Cache data for 24 hours
def get_cmu_program_data():
    """
    Scrapes a list of CMU Engineering pages using resilient, updated selectors.
    """
    base_url = "https://engineering.cmu.edu"
    # These URLs are confirmed to be the correct entry points for Graduate Studies
    source_urls = [
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
    ]
    
    all_programs = {}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    st.subheader("Data Fetch Status (Cached)")
    status_expander = st.expander("Show Data Scraping Log", expanded=False)

    with status_expander:
        progress_bar = st.progress(0, text="Starting live data fetch...")
        
        # --- PHASE 1: COLLECT PROGRAM LINKS ---
        st.markdown("---")
        st.caption("Phase 1: Collecting Program Links from Index Pages")
        for i, page_url in enumerate(source_urls):
            page_name = page_url.split('/')[-1]
            progress_bar.progress((i + 1) / len(source_urls) / 2, text=f"Scanning link sources: {page_name}")
            
            html_content = robust_request(page_url, headers)
            if not html_content:
                st.warning(f"Skipping page: {page_name} (Failed to retrieve content).")
                continue
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # --- UPDATED RESILIENT SELECTOR for main content area ---
            # Trying the standard <main> tag, common modern wrappers, and legacy IDs
            content_area = soup.find('main') or soup.find('div', class_='main-content') or soup.find('div', id='content') or soup.find('div', class_='content-wrapper') or soup.find('div', id='content_a')
            
            if not content_area:
                st.warning(f"Could not find main content area on {page_name}. Skipping link extraction.")
                continue
                
            # Filter links only within the main content area
            all_links = content_area.find_all('a', href=True)

            for link in all_links:
                program_name = link.text.strip()
                href = link.get('href', '').strip()
                
                # Intelligent filtering to find actual graduate programs
                is_program_link = ('M.S.' in program_name or 'Ph.D.' in program_name or 'Master' in program_name or 'Doctor' in program_name)
                # Ensure the link is not just a button/navigation and is a relevant path
                is_valid_path = ('/education/graduate-studies/' in href or '/departments/' in href) and not href.startswith('#')
                
                if is_program_link and is_valid_path and len(program_name) > 10: 
                    program_url = urljoin(base_url, href)
                    
                    degree_type = 'Other'
                    if 'M.S.' in program_name or 'Master' in program_name: degree_type = 'M.S.'
                    elif 'Ph.D.' in program_name or 'Doctor' in program_name: degree_type = 'Ph.D.'
                    
                    if program_url not in all_programs:
                        all_programs[program_url] = {'name': program_name, 'url': program_url, 'description': '', 'degree_type': degree_type}
            time.sleep(0.1) # Be polite to the server

        # --- PHASE 2: FETCH DESCRIPTIONS ---
        programs_list = list(all_programs.values())
        total_programs = len(programs_list)
        st.caption(f"Phase 2: Fetching details for {total_programs} unique programs.")
        
        for i, program in enumerate(programs_list):
            progress_bar.progress(0.5 + (i + 1) / total_programs / 2, text=f"Fetching details for: {program['name']}")
            
            sub_html = robust_request(program['url'], headers, timeout=10)
            if sub_html:
                sub_soup = BeautifulSoup(sub_html, 'html.parser')
                
                # --- UPDATED RESILIENT DESCRIPTION EXTRACTION ---
                # Search the entire body for a significant description block
                main_body = sub_soup.find('main') or sub_soup.find('div', class_='main-content') or sub_soup.body
                description_text = 'No detailed description found.'

                if main_body:
                    # Look for the first few paragraphs within the main body and join them
                    description_paragraphs = main_body.find_all('p', limit=4)
                    text_parts = [p.get_text(strip=True) for p in description_paragraphs if len(p.get_text(strip=True)) > 50] # Filter out short text like captions/nav
                    
                    if text_parts:
                        description_text = ' '.join(text_parts)
                        program['description'] = description_text[:700].rstrip() + '...' if len(description_text) > 700 else description_text
                    else:
                        program['description'] = 'No sufficiently detailed description found on page.'
                else:
                    program['description'] = 'Could not locate the main content body for description.'
            else:
                program['description'] = 'Could not retrieve program details.'
            time.sleep(0.1) # Be polite to the server
            
        progress_bar.empty()
        
    if not programs_list:
        st.error("Scraping finished, but no program data was collected. Please check the source URLs or website structure.")
        return pd.DataFrame()
        
    st.success(f"Successfully scraped and indexed {len(programs_list)} unique graduate programs.")
    return pd.DataFrame(programs_list)

# --- AI PROVIDER FUNCTIONS (No change needed here, keeping for completeness) ---

# General API Call for Embedding (Non-cached version with retry for robustness)
def _call_embedding_api(api_key: str, endpoint: str, payload: Dict[str, Any], model: str) -> Optional[np.ndarray]:
    for attempt in range(3):
        try:
            response = requests.post(endpoint, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=20)
            response.raise_for_status()
            return np.array(response.json()['data'][0]['embedding'])
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt) # Exponential backoff
                continue
            st.error(f"{model} Embedding API call failed after multiple retries: {e}")
            return None
        except Exception as e:
            st.error(f"{model} Embedding API processing failed: {e}")
            return None
    return None

@st.cache_data
def get_deepseek_embedding(text: str) -> Optional[np.ndarray]:
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
    if not api_key: return None
    payload = {"input": text, "model": "deepseek-embedding"}
    endpoint = "https://api.deepseek.com/v1/embeddings"
    return _call_embedding_api(api_key, endpoint, payload, "DeepSeek")

@st.cache_data
def get_gemini_embedding(text: str) -> Optional[np.ndarray]:
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        result = genai.embed_content(model="models/embedding-001", content=text, task_type="RETRIEVAL_QUERY")
        return np.array(result['embedding'])
    except Exception as e: 
        st.error(f"Gemini Embedding API call failed: {e}")
        return None

# General API Call for Analysis (Non-cached version with retry for robustness)
def _call_analysis_api(api_key: str, endpoint: str, payload: Dict[str, Any], model: str) -> str:
    for attempt in range(3):
        try:
            response = requests.post(endpoint, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt) # Exponential backoff
                continue
            return f"AI analysis unavailable: {model} API call failed after multiple retries: {e}"
        except Exception as e:
            return f"AI analysis unavailable: {model} API processing failed: {e}"
    return f"AI analysis unavailable: Unknown error."

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
    
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5, "max_tokens": 400}
    endpoint = "https://api.deepseek.com/v1/chat/completions"
    return _call_analysis_api(api_key, endpoint, payload, "DeepSeek")

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
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: 
        return f"AI analysis failed to generate (Gemini): {e}"

# --- MAIN APPLICATION LOGIC (No major logic change, only data dependency) ---
def main():
    st.title("🎓 CMU Engineering Program Navigator")
    st.markdown("Answer a few questions to discover the graduate program that best fits your academic and career goals.")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.image("https://www.cmu.edu/brand/brand-guidelines/assets/images/wordmarks-and-initials/cmu-wordmark-stacked-r-c.png", use_container_width=True)
        st.header("AI Configuration")
        
        available_providers = []
        if st.secrets.get("DEEPSEEK_API_KEY"): available_providers.append("DeepSeek")
        if st.secrets.get("GEMINI_API_KEY"): available_providers.append("Google Gemini")
        
        if not available_providers: 
            st.error("No AI provider API key found in app secrets. Please check your deployment secrets.")
            st.stop()
            
        ai_provider = st.selectbox("Choose AI Provider", available_providers) if len(available_providers) > 1 else available_providers[0]
        st.info(f"Using **{ai_provider}** API for analysis.")
        st.markdown("---")
        st.info("This tool is a proof-of-concept and not an official admissions tool.")

    # Select the correct functions based on user choice
    embedding_function = get_deepseek_embedding if ai_provider == "DeepSeek" else get_gemini_embedding
    analysis_function = get_deepseek_analysis if ai_provider == "DeepSeek" else get_gemini_analysis

    # Load and process data (runs once due to caching)
    # This will now use the updated resilient scraping logic
    df_programs = get_cmu_program_data()
    if df_programs.empty:
        st.warning("Program data could not be loaded. Cannot proceed with recommendations."); return

    # --- Interactive Questionnaire ---
    st.subheader("Tell us about yourself:")
    
    common_majors = ["Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", 
                    "Chemical Engineering", "Biomedical Engineering", "Materials Science", "Physics", 
                    "Mathematics", "Industrial Design/Art", "Other Engineering"]
    
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
        # --- Synthesize User Query ---
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
            st.code(synthesized_query, language='text')
        
        # --- Embeddings & Matching ---
        
        # We need to re-index only if the AI provider changes (forcing a rerun of the embedding function)
        embedding_key = f'embedding__{ai_provider}'
        if embedding_key not in st.session_state:
            with st.spinner(f"🧠 Indexing programs using {ai_provider}... (This runs once and is cached)"):
                df_programs['embedding'] = df_programs.apply(lambda row: embedding_function(f"Program: {row['name']}\nDescription: {row['description']}"), axis=1)
                df_programs.dropna(subset=['embedding'], inplace=True)
                st.session_state[embedding_key] = df_programs # Store the indexed DataFrame
        
        df = st.session_state[embedding_key].copy()
        
        with st.spinner(f"🔍 Analyzing your profile and finding the best matches with {ai_provider}..."):
            query_embedding = embedding_function(synthesized_query)
            
            if query_embedding is not None and not df.empty:
                # Calculate cosine similarity
                df['similarity'] = df['embedding'].apply(
                    lambda x: np.dot(x, query_embedding) / (np.linalg.norm(x) * np.linalg.norm(query_embedding))
                )
                
                # Boost score for degree type match
                df['degree_match_boost'] = df.apply(lambda row: 0.1 if row['degree_type'] == degree_level else 0, axis=1)
                df['final_score'] = df['similarity'] + df['degree_match_boost']
                
                results = df.sort_values('final_score', ascending=False).head(3)

                st.subheader(f"Top 3 Program Recommendations for a {degree_level} Applicant")
                if results.empty:
                    st.warning("No strong matches found. Try adjusting your criteria.")
                else:
                    for i, (_, program) in enumerate(results.iterrows()):
                        st.markdown("---"); st.markdown(f"### **{i+1}. {program['name']}**")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1: 
                            # Display similarity as a progress bar
                            st.progress(program['similarity'], text=f"**Profile Match Score: {program['similarity']:.1%}**")
                        with col2: 
                            st.link_button("Go to Program Website ↗️", program['url'], use_container_width=True)
                        
                        with st.expander("**Program Overview from CMU Website**"): 
                            st.write(program['description'])
                        
                        with st.spinner("🤖 Generating personalized AI Advisor analysis..."):
                            analysis = analysis_function(program['name'], program['description'], synthesized_query)
                        
                        st.markdown("**AI-Powered Advisor Analysis**"); 
                        st.info(analysis)
            else:
                st.error("Could not process your query or the program data index is empty.")

if __name__ == "__main__":
    main()
