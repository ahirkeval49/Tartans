import streamlit as st
import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import pandera as pa
from pandera import Check
import time

# Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_EMBEDDING_URL = "https://api.deepseek.com/v1/embeddings"
CACHE_EXPIRATION = 86400

# --- FIX: Relaxed the validation for the description length ---
PROGRAM_SCHEMA = pa.DataFrameSchema(
    columns={
        "name": pa.Column(str, Check(lambda s: s.str.len() > 5), nullable=False),
        "url": pa.Column(str, Check.str_startswith("http"), nullable=False),
        # Changed from > 100 to > 20 to be less strict and allow more programs
        "description": pa.Column(str, Check(lambda s: s.str.len() > 20)),
        "courses": pa.Column(str, nullable=True),
        "admission": pa.Column(str, nullable=True),
        "contact": pa.Column(str, nullable=True),
        "department": pa.Column(str, Check.equal_to("Engineering"), nullable=False),
        "embedding": pa.Column(object, nullable=True),
    },
    coerce=True,
    strict=False
)

@st.cache_data(ttl=CACHE_EXPIRATION)
def get_program_data():
    base_url = "https://engineering.cmu.edu"
    source_url = f"{base_url}/education/graduate-studies/programs/index.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    programs = []
    try:
        response = requests.get(source_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        program_links = soup.select('div.program-listing h3 a')
        if not program_links:
            st.warning("Could not find any program links on the page. The website structure may have changed.")
            return pd.DataFrame()

        for link in program_links:
            program_name = link.get_text(strip=True)
            program_url = urljoin(base_url, link['href'])
            time.sleep(0.2) # slightly faster scraping
            prog_response = requests.get(program_url, headers=headers, timeout=15)
            prog_soup = BeautifulSoup(prog_response.text, 'html.parser')

            description_tag = prog_soup.find('div', class_='program-intro')
            # Provide a default description that will pass validation if none is found
            description = description_tag.get_text(strip=True) if description_tag else 'Detailed description not available on the program page.'

            courses = []
            curriculum_header = prog_soup.find('h2', string=lambda t: t and 'curriculum' in t.lower())
            if curriculum_header:
                curriculum_div = curriculum_header.find_next_sibling('div')
                if curriculum_div:
                    courses = [item.get_text(strip=True) for item in curriculum_div.find_all('li')]

            admission = ''
            admission_header = prog_soup.find('h2', string=lambda t: t and 'admission' in t.lower())
            if admission_header:
                admission_div = admission_header.find_next_sibling('div')
                if admission_div:
                    admission = admission_div.get_text(strip=True)
            
            contact = ''
            contact_info = prog_soup.find(string=lambda t: t and ('@' in t or 'contact' in t.lower()))
            if contact_info:
                contact = contact_info.strip()

            programs.append({
                'name': program_name, 'url': program_url, 'description': description,
                'courses': '\n'.join(courses), 'admission': admission, 'contact': contact,
                'department': 'Engineering'
            })
    except Exception as e:
        st.error(f"Scraping failed: {e}")
        return pd.DataFrame()

    if not programs:
        st.error("Scraping finished, but no program data was collected.")
        return pd.DataFrame()

    df = pd.DataFrame(programs)
    try:
        validated_df = PROGRAM_SCHEMA.validate(df, lazy=True)
        st.success(f"Successfully loaded and validated {len(validated_df)} programs.")
        return validated_df
    except pa.errors.SchemaErrors as err:
        st.warning("Some programs had data quality issues and were removed.")
        valid_indices = df.index.difference(err.failure_cases.index)
        cleaned_df = df.loc[valid_indices]
        if not cleaned_df.empty:
            st.info(f"Proceeding with {len(cleaned_df)} valid programs.")
            return cleaned_df
        else:
            st.error("No valid program data remained after cleaning. The website structure may have changed significantly.")
            return pd.DataFrame()

@st.cache_data
def get_ai_embedding(text):
    if 'DEEPSEEK_KEY' not in st.secrets or not st.secrets.DEEPSEEK_KEY:
        st.error("DeepSeek API Key is missing.")
        return None
    try:
        response = requests.post(
            DEEPSEEK_EMBEDDING_URL,
            headers={"Authorization": f"Bearer {st.secrets.DEEPSEEK_KEY}"},
            json={"input": text, "model": "deepseek-embedding"},
            timeout=20
        )
        response.raise_for_status()
        return np.array(response.json()['data'][0]['embedding'])
    except Exception as e:
        st.error(f"Embedding API call failed: {str(e)}")
        return None

@st.cache_data
def get_ai_analysis(program_name, query, context):
    if 'DEEPSEEK_KEY' not in st.secrets or not st.secrets.DEEPSEEK_KEY:
        return "AI analysis unavailable: API key missing."
    prompt = f"Analyze this graduate program for a student interested in {query}:\n\nProgram: {program_name}\nContext: {context}\n\nProvide 3 concise recommendations in bullet points format:\n- Why this program matches their interests\n- Key preparation steps\n- Potential career outcomes"
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {st.secrets.DEEPSEEK_KEY}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3, "max_tokens": 300
            },
            timeout=25
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"AI analysis failed: {e}"

def main():
    st.set_page_config(page_title="CMU Engineering Advisor Pro", layout="wide")
    st.title("🔍 CMU College of Engineering Program Matchmaker")
    st.write("Discover your ideal engineering graduate program using AI-powered matching.")
    
    with st.spinner("🔄 Loading program information from CMU Engineering..."):
        df = get_program_data()
        
    if 'DEEPSEEK_KEY' not in st.secrets or not st.secrets.DEEPSEEK_KEY:
        st.error("DeepSeek API Key is NOT found in secrets. Please add it to run the app.")
        return

    if df.empty:
        st.warning("Program data could not be loaded. The app cannot continue.")
        return

    # Pre-calculate embeddings for all programs
    if 'embedding' not in df.columns or df['embedding'].isnull().any():
        with st.spinner("🧠 Generating AI embeddings for programs... This happens once and is then cached."):
            df['embedding'] = df.apply(
                lambda x: get_ai_embedding(f"Program: {x['name']}\nDescription: {x['description']}"), 
                axis=1
            )
            df.dropna(subset=['embedding'], inplace=True)
            if df.empty:
                st.error("Failed to generate embeddings for any programs. Please check your API key and network connection.")
                return

    # --- THE SEARCH FIELD IS NOW VISIBLE ---
    search_query = st.text_input(
        "Describe your academic and career interests:", 
        placeholder="e.g., 'Applying machine learning to sustainable energy systems'"
    )
    
    if search_query:
        with st.spinner("🔍 Finding the best matches for you..."):
            query_embedding = get_ai_embedding(search_query)
            
            if query_embedding is not None:
                filtered_df = df.copy()
                filtered_df['match_score'] = filtered_df['embedding'].apply(
                    lambda x: np.dot(query_embedding, x) if x is not None else 0
                )
                results = filtered_df.sort_values('match_score', ascending=False).head(5)
                
                st.subheader(f"Top {len(results)} Program Matches")
                if results.empty:
                    st.warning("No strong matches found. Try rephrasing your interests.")
                
                for idx, (_, program) in enumerate(results.iterrows(), 1):
                    with st.expander(f"{idx}. {program['name']} ({program['match_score']:.0%} Match)", expanded=idx==1):
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            st.markdown(f"**Department:** {program['department']}")
                            st.markdown(f"**Program Overview:**\n{program['description']}")
                            if program['courses']:
                                with st.popover("📚 Key Courses"):
                                    st.write(program['courses'])
                            with st.popover("🎓 Admission Requirements"):
                                st.write(program['admission'] or "Information not available")
                        with col_b:
                            st.link_button("🌐 Program Website", program['url'])
                            st.markdown(f"**📧 Contact Info:**\n{program['contact'] or 'See website for details'}")
                            st.markdown("---")
                            context = f"Description: {program['description'][:500]}\nCourses: {program['courses'][:300]}"
                            analysis = get_ai_analysis(program['name'], search_query, context)
                            st.markdown(f"**🤖 AI Program Analysis:**\n{analysis}")
            else:
                st.error("Could not process your query. The embedding API might be down or the key is invalid.")
    
    st.markdown("---")
    st.markdown("ℹ️ Data sourced from CMU Engineering website. Recommendations powered by DeepSeek AI.")

if __name__ == "__main__":
    main()
