# tartans.py
# CMU College of Engineering Program Navigator
# Author: Gemini (Google AI)
# Date: October 13, 2025 (Updated for Maximum Scraping Resilience - Version 3.0)

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
    Scrapes a list of CMU Engineering pages by treating each department URL 
    as a program source and inferring M.S. and Ph.D. programs from it.
    This avoids unreliable link-finding.
    """
    base_url = "https://engineering.cmu.edu"
    
    # These URLs are the department-level program pages we must scrape for content.
    department_urls = [
        "https://engineering.cmu.edu/education/graduate-studies/programs/bme.html", # Biomedical Engineering
        "https://engineering.cmu.edu/education/graduate-studies/programs/cheme.html", # Chemical Engineering
        "https://engineering.cmu.edu/education/graduate-studies/programs/cee.html", # Civil & Environmental Engineering
        "https://engineering.cmu.edu/education/graduate-studies/programs/ece.html", # Electrical & Computer Engineering
        "https://engineering.cmu.edu/education/graduate-studies/programs/epp.html", # Engineering & Public Policy
        "https://engineering.cmu.edu/education/graduate-studies/programs/ini.html", # Information Networking Institute
        "https://engineering.cmu.edu/education/graduate-studies/programs/iii.html", # Integrated Innovation Institute
        "https://engineering.cmu.edu/education/graduate-studies/programs/mse.html", # Materials Science & Engineering
        "https://engineering.cmu.edu/education/graduate-studies/programs/meche.html", # Mechanical Engineering
        "https://engineering.cmu.edu/education/graduate-studies/programs/cmu-africa.html", # CMU Africa
        "https://engineering.cmu.edu/education/graduate-studies/programs/sv.html" # Silicon Valley
    ]
    
    programs_list = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    st.subheader("Data Fetch Status (Cached)")
    status_expander = st.expander("Show Data Scraping Log", expanded=False)

    with status_expander:
        progress_bar = st.progress(0, text="Starting direct program content fetch...")
        
        # --- PHASE 1: DIRECTLY FETCH CONTENT FROM DEPARTMENT PAGES ---
        st.markdown("---")
        st.caption("Phase 1: Fetching content directly from departmental program pages.")
        
        for i, page_url in enumerate(department_urls):
            page_name = page_url.split('/')[-1]
            progress_bar.progress((i + 1) / len(department_urls), text=f"Processing department page: {page_name}")
            
            html_content = robust_request(page_url, headers)
            if not html_content:
                st.warning(f"Skipping page: {page_name} (Failed to retrieve content).")
                continue
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 1. EXTRACT PROGRAM NAME (More robust: look for H1, then H2, then fallback to title)
            program_name_base = "CMU Program"
            if soup.find('h1'):
                program_name_base = soup.find('h1').get_text(strip=True).replace("Graduate", "").strip()
            elif soup.find('title'):
                program_name_base = soup.find('title').get_text(strip=True).split('|')[0].strip()
                
            # 2. AGGRESSIVE DESCRIPTION EXTRACTION
            description_text = 'No sufficiently detailed description found on page.'
            description_paragraphs = soup.body.find_all('p', limit=10) if soup.body else []
            # Only use paragraphs longer than 50 characters to avoid footers/nav links
            text_parts = [p.get_text(strip=True) for p in description_paragraphs if len(p.get_text(strip=True)) > 50]
            
            if text_parts:
                description_text = ' '.join(text_parts)
                # Limit and clean up the description
                description_text = description_text[:700].rstrip() + '...' if len(description_text) > 700 else description_text
            
            # 3. CREATE TWO PROGRAM ENTRIES (M.S. and Ph.D.) for the matching algorithm
            
            # M.S. Program Entry
            ms_name = f"Master of Science ({program_name_base})"
            programs_list.append({
                'name': ms_name, 
                'url': page_url, 
                'description': description_text, 
                'degree_type': 'M.S.'
            })
            
            # Ph.D. Program Entry
            phd_name = f"Doctor of Philosophy ({program_name_base})"
            programs_list.append({
                'name': phd_name, 
                'url': page_url, 
                'description': description_text, 
                'degree_type': 'Ph.D.'
            })
            
            st.info(f"Successfully indexed two programs from: {page_name} (M.S. and Ph.D. in {program_name_base})")
            
            time.sleep(0.1) # Be polite to the server

        # --- FINALIZATION ---
        progress_bar.empty()
        
    if not programs_list:
        st.error("Scraping finished, but no program data was collected. Please check the source URLs or website structure.")
        return pd.DataFrame()
        
    st.success(f"Successfully scraped and indexed {len(programs_list)} distinct graduate program variants.")
    return pd.DataFrame(programs_list)

# --- AI PROVIDER FUNCTIONS (Updated DeepSeek Embedding) ---

def _call_embedding_api(api_key: str, endpoint: str, payload: Dict[str, Any], model: str) -> Optional[np.ndarray]:
    for attempt in range(3):
        try:
            response = requests.post(endpoint, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=20)
            response.raise_for_status()
            return np.array(response.json()['data'][0]['embedding'])
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            # Note: This is where the 404 error was logged for DeepSeek
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
    # FIX: Correcting the payload:
    # 1. The 'input' must be a list of strings (even if it's just one).
    # 2. The 'model' name must be the correct identifier for the DeepSeek embedding model.
    payload = {"input": [text], "model": "deepseek-embed"} 
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

def _call_analysis_api(api_key: str, endpoint: str, payload: Dict[str, Any], model: str) -> str:
    for attempt in range(3):
        try:
            response = requests.post(endpoint, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
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

# --- MAIN APPLICATION LOGIC ---
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
        
        embedding_key = f'embedding__{ai_provider}'
        if embedding_key not in st.session_state:
            with st.spinner(f"🧠 Indexing programs using {ai_provider}... (This runs once and is cached)"):
                df_programs['embedding'] = df_programs.apply(lambda row: embedding_function(f"Program: {row['name']}\nDescription: {row['description']}"), axis=1)
                df_programs.dropna(subset=['embedding'], inplace=True)
                st.session_state[embedding_key] = df_programs 
        
        df = st.session_state[embedding_key].copy()
        
        with st.spinner(f"🔍 Analyzing your profile and finding the best matches with {ai_provider}..."):
            query_embedding = embedding_function(synthesized_query)
            
            if query_embedding is not None and not df.empty:
                # Calculate cosine similarity
                df['similarity'] = df['embedding'].apply(
                    lambda x: np.dot(x, query_embedding) / (np.linalg.norm(x) * np.linalg.norm(query_embedding))
                )
                
                # Boost score for degree type match
                # Only keep programs that match the selected degree level
                df_filtered = df[df['degree_type'] == degree_level].copy()
                
                if df_filtered.empty:
                    st.warning(f"No programs matching the required '{degree_level}' degree level were found in the index.")
                    return
                
                results = df_filtered.sort_values('similarity', ascending=False).head(3)

                st.subheader(f"Top 3 Program Recommendations for a {degree_level} Applicant")
                if results.empty:
                    st.warning("No strong matches found. Try adjusting your criteria.")
                else:
                    for i, (_, program) in enumerate(results.iterrows()):
                        st.markdown("---"); st.markdown(f"### **{i+1}. {program['name']}**")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1: 
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
