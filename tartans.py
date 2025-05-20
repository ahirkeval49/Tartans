import streamlit as st
import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_EMBEDDING_URL = "https://api.deepseek.com/v1/embeddings"

# Web Scraper
def get_program_links():
    base_url = "https://engineering.cmu.edu"
    source_url = "https://engineering.cmu.edu/education/graduate-studies/programs/index.html"
    
    try:
        response = requests.get(source_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        return [
            (li.a.text.strip(), f"{base_url}{li.a['href']}")
            for li in soup.select('div.content-asset li')
            if li.a and li.a.get('href')
        ]
    except Exception as e:
        st.error(f"Failed to load programs: {str(e)}")
        return []

# Program Detail Extractor
def get_program_details(url):
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        return {
            'description': ' '.join(p.text for p in soup.select('div.col-lg-8 p')),
            'courses': '\n'.join(soup.find('h2', string=lambda t: 'curriculum' in t.lower()).find_next_sibling().stripped_strings) if soup.find('h2', string=lambda t: 'curriculum' in t.lower()) else '',
            'admission': soup.find('h2', string='Admission Requirements').find_next_sibling().get_text(' ') if soup.find('h2', string='Admission Requirements') else '',
            'contact': soup.find('h2', string='Contact Us').find_next_sibling().get_text('\n') if soup.find('h2', string='Contact Us') else ''
        }
    except Exception as e:
        st.error(f"Error loading {url}: {str(e)}")
        return None

# AI Integration
@st.cache_data
def get_ai_response(prompt):
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {st.secrets.DEEPSEEK_KEY}"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.4}
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        st.error("AI service unavailable. Try again later.")
        return ""

@st.cache_data
def get_ai_embedding(text):
    try:
        response = requests.post(
            DEEPSEEK_EMBEDDING_URL,
            headers={"Authorization": f"Bearer {st.secrets.DEEPSEEK_KEY}"},
            json={"input": text, "model": "deepseek-embedding"}
        )
        return response.json()['data'][0]['embedding']
    except:
        return None

# Core Application
def main():
    st.title("CMU Engineering Program Finder")
    st.subheader("Search Graduate Programs by Interest Area")
    
    # Load data
    with st.spinner("Loading program information..."):
        programs = [
            {**{'name': name, 'url': url}, **get_program_details(url)} 
            for name, url in get_program_links()
            if get_program_details(url)
        ]
        df = pd.DataFrame(programs)
        df['embedding'] = df.apply(lambda x: get_ai_embedding(f"{x['name']} {x['description']} {x['courses']}"), axis=1)
    
    # Search interface
    query = st.text_input("What are you interested in studying?")
    if not query:
        return
    
    # Process query
    with st.spinner("Finding best matches..."):
        query_embed = get_ai_embedding(query)
        df['score'] = df['embedding'].apply(lambda x: cosine_similarity([query_embed], [x])[0][0] if x else 0)
        results = df.sort_values('score', ascending=False).head(3)
    
    # Display results
    for _, row in results.iterrows():
        with st.expander(f"🎓 {row['name']} (Relevance: {row['score']:.0%})"):
            st.markdown(f"**Description:** {row['description'][:400]}...")
            st.markdown(f"**Sample Courses:**\n{row['courses'][:300]}...")
            st.markdown(f"**Admission Requirements:**\n{row['admission'][:500]}...")
            st.markdown(f"[Program Website]({row['url']}) | **Contact:** {row['contact'].split('\n')[0]}")
            
            advice = get_ai_response(f"Suggest preparation steps for {row['name']} focusing on {query}")
            st.markdown(f"**AI Recommendation:**\n{advice}")

if __name__ == "__main__":
    main()
