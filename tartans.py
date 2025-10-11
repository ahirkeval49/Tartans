import streamlit as st
import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import pandera as pa
from pandera import Check 
import time # NEW: Added for polite scraping (throttling)

# Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_EMBEDDING_URL = "https://api.deepseek.com/v1/embeddings"
CACHE_EXPIRATION = 86400  # 24 hours in seconds

# NEW: Define the explicit schema for College of Engineering program data
# This ensures data quality and resilience against scraping errors
PROGRAM_SCHEMA = pa.DataFrameSchema(
    columns={
        "name": pa.Column(str, Check(lambda s: s.str.len() > 5), nullable=False), # Program name must be substantial
        "url": pa.Column(str, Check.str_startswith("http"), nullable=False), # Must be a valid URL
        "description": pa.Column(str, Check(lambda s: s.str.len() > 100)), # Description must be long enough to be useful
        "courses": pa.Column(str, nullable=True),
        "admission": pa.Column(str, nullable=True),
        "contact": pa.Column(str, nullable=True),
        "department": pa.Column(str, Check.equal_to("Engineering"), nullable=False), # Explicitly check for CoE
        "embedding": pa.Column(object, nullable=True), # Placeholder for the embedding list/array
    },
    coerce=True,
    strict=False # Allow for potential future columns, but strictly check the ones defined
)

@st.cache_data(ttl=CACHE_EXPIRATION)
def get_program_data():
    base_url = "https://engineering.cmu.edu"
    source_url = f"{base_url}/education/graduate-studies/programs/index.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }

    programs =
    try:
        response = requests.get(source_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Locate all program links (specific to CoE structure)
        program_links = soup.select('div.accordion-item.program-listing a.button')
        
        st.info(f"Found {len(program_links)} potential programs. Fetching details...")

        for link in program_links:
            program_name = link.get_text(strip=True)
            program_url = urljoin(base_url, link['href'])

            # Be polite: Wait a short time between requests to individual pages
            time.sleep(0.5) 

            # Fetch program-specific page
            prog_response = requests.get(program_url, headers=headers, timeout=15)
            prog_soup = BeautifulSoup(prog_response.text, 'html.parser')

            # Extract description
            description_tag = prog_soup.find('div', class_='program-intro')
            description = description_tag.get_text(strip=True) if description_tag else ''

            # Extract courses
            courses =
            curriculum_header = prog_soup.find('h2', string=lambda t: t and 'curriculum' in t.lower())
            if curriculum_header:
                curriculum_div = curriculum_header.find_next_sibling('div')
                if curriculum_div:
                    courses =

            # Extract admission requirements
            admission = ''
            admission_header = prog_soup.find('h2', string=lambda t: t and 'admission' in t.lower())
            if admission_header:
                admission_div = admission_header.find_next_sibling('div')
                if admission_div:
                    admission = admission_div.get_text(strip=True)

            # Extract contact information
            contact = ''
            # Search broadly for contact details, prioritizing emails/contact phrases
            contact_info = prog_soup.find(string=lambda t: t and ('@' in t or 'contact' in t.lower()))
            if contact_info:
                contact = contact_info.strip()

            programs.append({
                'name': program_name,
                'url': program_url,
                'description': description,
                'courses': '\n'.join(courses),
                'admission': admission,
                'contact': contact,
                'department': 'Engineering' 
            })

    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error during scraping: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"General scraping failure: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(programs)
    
    # NEW: Validate and clean data using the defined schema
    try:
        # Validate and coerce types, dropping records that fail schema checks
        df = PROGRAM_SCHEMA.validate(df, lazy=True).dropna(subset=['name', 'description'])
        st.success(f"Successfully validated and prepared {len(df)} College of Engineering programs.")
    except pa.errors.SchemaError as err:
        st.warning(f"Data Quality Warning: Detected and removed invalid data (Error: {err.message}).")
        # For simplicity, we filter out bad records if validation fails
        df = df[~df.index.isin(err.failure_cases.index)]

    return df


@st.cache_data
def get_ai_embedding(text):
    # This remains the same
    try:
        response = requests.post(
            DEEPSEEK_EMBEDDING_URL,
            headers={"Authorization": f"Bearer {st.secrets.DEEPSEEK_KEY}"},
            json={"input": text, "model": "deepseek-embedding"},
            timeout=20
        )
        # Ensure the response is successful and contains data
        response.raise_for_status()
        return np.array(response.json()['data']['embedding'])
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return None

@st.cache_data
def get_ai_analysis(program_name, query, context):
    # This remains the same
    prompt = f"""Analyze this graduate program for a student interested in {query}:
    
    Program: {program_name}
    Context: {context}
    
    Provide 3 concise recommendations in bullet points format:
    - Why this program matches their interests
    - Key preparation steps
    - Potential career outcomes
    """
    
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {st.secrets.DEEPSEEK_KEY}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 300
            },
            timeout=25
        )
        response.raise_for_status()
        return response.json()['choices']['message']['content']
    except:
        return "AI analysis currently unavailable"

def main():
    st.set_page_config(page_title="CMU Engineering Advisor Pro", layout="wide")
    st.title("🔍 CMU College of Engineering Program Matchmaker")
    st.write("Discover your ideal engineering graduate program using AI-powered matching")
    
    # Load data
    with st.spinner("🔄 Loading latest program information from CMU Engineering..."):
        df = get_program_data()
        
    # Check for DeepSeek key availability (optional safety check)
    if 'DEEPSEEK_KEY' not in st.secrets:
        st.error("DeepSeek API Key is required in st.secrets.")
        return

    if not df.empty:
        # Calculate embeddings if not already present (only happens on first run due to caching)
        if 'embedding' not in df.columns or df['embedding'].isnull().any():
             df['embedding'] = df.apply(
                lambda x: get_ai_embedding(f"Program: {x['name']}\nDescription: {x['description']}\nCourses: {x['courses']}"), 
                axis=1
            )
             df.dropna(subset=['embedding'], inplace=True) # Remove rows where embedding generation failed

    
    # Search interface
    col1, col2 = st.columns([1, 2])
    with col1:
        search_query = st.text_input("Describe your academic/career interests:", 
                                   placeholder="e.g., 'Applying machine learning to sustainable energy systems'")
    
    # The department filter section has been removed as per the user's explicit request 
    # to focus only on the College of Engineering data (which is filtered in get_program_data).
    
    if search_query and not df.empty:
        with st.spinner("🔍 Finding best matches..."):
            query_embedding = get_ai_embedding(search_query)
            if query_embedding is not None:
                
                # Filter dataframe (no departmental filter needed, already CoE)
                filtered_df = df
                
                # Calculate similarity: Using numpy.dot for efficiency on normalized vectors
                filtered_df['match_score'] = filtered_df['embedding'].apply(
                    lambda x: np.dot(query_embedding, x) if x is not None else 0
                )
                results = filtered_df.sort_values('match_score', ascending=False).head(5)
                
                # Display results
                st.subheader(f"Top {len(results)} Program Matches")
                for idx, (_, program) in enumerate(results.iterrows(), 1):
                    with st.expander(f"{idx}. 🥇 {program['name']} ({program['match_score']:.0%} Match)", expanded=idx==1):
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
                            
                            # AI Analysis Section
                            st.markdown("---")
                            # Note: Context injection limits remain to avoid excessive token usage 
                            # in the cached function, but this should be revisited architecturally.
                            context = f"Description: {program['description'][:500]}\nCourses: {program['courses'][:300]}"
                            analysis = get_ai_analysis(program['name'], search_query, context)
                            st.markdown(f"**🤖 AI Program Analysis:**\n{analysis}")
            else:
                st.error("Error processing your query. Please check API Key and network connection.")
    
    # Add footer
    st.markdown("---")
    st.markdown("ℹ️ Data sourced from CMU Engineering website. Recommendations powered by DeepSeek AI.")

if __name__ == "__main__":
    main()
