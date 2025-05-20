import streamlit as st
import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Deepseek API configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_EMBEDDING_URL = "https://api.deepseek.com/v1/embeddings"

# Web Scraping Functions
def get_program_links():
    url = "https://engineering.cmu.edu/education/graduate-studies/programs/index.html","https://engineering.cmu.edu/education/graduate-studies/programs/index.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    programs = []
    main_content = soup.find('div', class_='content-asset')
    for item in main_content.find_all('li'):
        link = item.find('a')
        if link:
            program_name = link.text.strip()
            program_url = "https://engineering.cmu.edu","https://engineering.cmu.edu/education/graduate-studies/programs/index.html"
            programs.append((program_name, program_url))
    return programs

def get_program_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    details = {
        'description': '',
        'courses': '',
        'research_topics': '',
        'admission_requirements': '',
        'contact_info': ''
    }
    
    # Extract main content
    main_content = soup.find('div', class_='col-12 col-lg-8')
    if main_content:
        details['description'] = ' '.join(p.text for p in main_content.find_all('p'))
        
        # Extract courses and research topics using heuristic patterns
        for h2 in main_content.find_all('h2'):
            if 'curriculum' in h2.text.lower():
                details['courses'] = ' '.join(h2.find_next_sibling().stripped_strings)
            if 'research' in h2.text.lower():
                details['research_topics'] = ' '.join(h2.find_next_sibling().stripped_strings)
    
    # Extract admission requirements
    admission_section = soup.find('h2', string='Admission Requirements')
    if admission_section:
        details['admission_requirements'] = ' '.join(
            admission_section.find_next_sibling().stripped_strings
        )
    
    # Extract contact info
    contact_section = soup.find('h2', string='Contact Us')
    if contact_section:
        details['contact_info'] = contact_section.find_next_sibling().get_text(separator='\n')
    
    return details

# Deepseek AI Integration
def get_deepseek_response(prompt, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }
    
    response = requests.post(DEEPSEEK_API_URL, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    return None

def get_deepseek_embedding(text, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "input": text,
        "model": "deepseek-embedding"
    }
    
    response = requests.post(DEEPSEEK_EMBEDDING_URL, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    return None

# Data Processing
@st.cache_data
def load_data(api_key):
    programs = get_program_links()
    data = []
    
    for name, url in programs:
        details = get_program_details(url)
        search_content = f"{name} {details['description']} {details['courses']} {details['research_topics']}"
        
        # Get Deepseek embedding
        embedding = get_deepseek_embedding(search_content, api_key)
        
        data.append({
            'Program Name': name,
            'Description': details['description'],
            'Courses': details['courses'],
            'Research Topics': details['research_topics'],
            'Admission Requirements': details['admission_requirements'],
            'Contact Info': details['contact_info'],
            'Program URL': url,
            'Search Content': search_content,
            'Embedding': embedding
        })
    
    return pd.DataFrame(data)

# Search Engine
def create_search_index(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Search Content'])
    return tfidf, tfidf_matrix

# Streamlit UI
st.title("CMU Engineering AI Program Advisor")
st.subheader("Intelligent Graduate Program Finder with Deepseek AI")

# Authentication
api_key = st.sidebar.text_input("Deepseek API Key", type="password")

if not api_key:
    st.warning("Please enter your Deepseek API key to continue")
    st.stop()

# Load data with progress
with st.spinner("Loading program data and AI models..."):
    df = load_data(api_key)
    tfidf, tfidf_matrix = create_search_index(df)

# Search Interface
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Ask about programs (e.g., 'AI programs with robotics courses'):")

with col2:
    search_mode = st.selectbox("Search Mode", ["Hybrid", "Semantic", "Keyword"])

# Process query
if query:
    with st.spinner("Analyzing your query with Deepseek AI..."):
        # Get Deepseek query expansion
        expanded_query = get_deepseek_response(
            f"Expand this academic program search query with related terms: {query}",
            api_key
        )
        
        # Get embeddings
        query_embedding = get_deepseek_embedding(query + " " + expanded_query, api_key)
        tfidf_scores = cosine_similarity(tfidf.transform([query]), tfidf_matrix)[0]
        
        if query_embedding:
            embedding_scores = cosine_similarity(
                [query_embedding],
                np.stack(df['Embedding'])
            )[0]
            
            # Combine scores based on search mode
            if search_mode == "Hybrid":
                combined_scores = 0.6 * embedding_scores + 0.4 * tfidf_scores
            elif search_mode == "Semantic":
                combined_scores = embedding_scores
            else:
                combined_scores = tfidf_scores
            
            df['score'] = combined_scores
            results = df.sort_values('score', ascending=False).head(5)
            
            # Generate AI summary
            ai_analysis = get_deepseek_response(
                f"Analyze these engineering programs {results[['Program Name', 'Description']].to_dict()} "
                f"for query '{query}'. Provide 2-3 sentence summary.",
                api_key
            )
            
            st.subheader("AI Analysis")
            st.markdown(f"*{ai_analysis}*")

else:
    results = df

# Display results
st.subheader("Matching Programs")
for _, row in results.iterrows():
    with st.expander(f"**{row['Program Name']}** (Score: {row['score']:.2f})"):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Description:** {row['Description'][:500]}...")
            if row['Courses']:
                st.markdown(f"**Key Courses:** {row['Courses'][:300]}...")
            if row['Research Topics']:
                st.markdown(f"**Research Areas:** {row['Research Topics'][:300]}...")
            st.markdown(f"**Admission Requirements:** {row['Admission Requirements'][:300]}...")
        with col2:
            st.markdown(f"[Program Website]({row['Program URL']})")
            st.markdown(f"**Contact:**\n{row['Contact Info']}")
            if query_embedding:
                similar_courses = get_deepseek_response(
                    f"Suggest related courses to {query} in {row['Program Name']}",
                    api_key
                )
                st.markdown(f"**AI Course Suggestions:**\n{similar_courses}")

# Add footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}
</style>
<div class="footer">
<p>Powered by Deepseek AI | Data sourced from CMU Engineering | Real-time AI analysis</p>
</div>
""", unsafe_allow_html=True)
