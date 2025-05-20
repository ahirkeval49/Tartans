import streamlit as st
import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_EMBEDDING_URL = "https://api.deepseek.com/v1/embeddings"

# Web Scraper with Error Handling
def get_program_data():
    try:
        base_url = "https://engineering.cmu.edu"
        source_url = "https://engineering.cmu.edu/education/graduate-studies/programs/index.html"
        
        response = requests.get(source_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        programs = []
        for li in soup.select('div.content-asset li'):
            if not li.a or not li.a.get('href'):
                continue
                
            program_url = f"{base_url}{li.a['href']}" if li.a['href'].startswith('/') else li.a['href']
            
            try:
                program_response = requests.get(program_url, timeout=15)
                program_soup = BeautifulSoup(program_response.text, 'html.parser')
                
                programs.append({
                    'name': li.a.text.strip(),
                    'url': program_url,
                    'description': ' '.join(p.text for p in program_soup.select('div.col-lg-8 p')),
                    'courses': '\n'.join(program_soup.find('h2', string=lambda t: 'curriculum' in t.lower())
                                       .find_next_sibling().stripped_strings) if program_soup.find('h2', string=lambda t: 'curriculum' in t.lower()) else '',
                    'admission': program_soup.find('h2', string='Admission Requirements')
                                   .find_next_sibling().get_text(' ') if program_soup.find('h2', string='Admission Requirements') else '',
                    'contact': program_soup.find('h2', string='Contact Us')
                                 .find_next_sibling().get_text('\n') if program_soup.find('h2', string='Contact Us') else ''
                })
            except Exception as e:
                st.error(f"Couldn't load {li.a.text.strip()}: {str(e)}")
        
        return pd.DataFrame(programs)
    
    except Exception as e:
        st.error(f"Failed to load program data: {str(e)}")
        return pd.DataFrame()

# AI Services with Secure API Access
@st.cache_data
def get_ai_embedding(text):
    try:
        response = requests.post(
            DEEPSEEK_EMBEDDING_URL,
            headers={"Authorization": f"Bearer {st.secrets.DEEPSEEK_KEY}"},
            json={"input": text, "model": "deepseek-embedding"},
            timeout=20
        )
        return response.json()['data'][0]['embedding']
    except:
        return None

@st.cache_data
def get_ai_advice(program_name, interest):
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {st.secrets.DEEPSEEK_KEY}"},
            json={
                "model": "deepseek-chat",
                "messages": [{
                    "role": "user", 
                    "content": f"Suggest preparation steps for {program_name} focusing on {interest}. Keep response under 100 words."
                }],
                "temperature": 0.4
            },
            timeout=25
        )
        return response.json()['choices'][0]['message']['content']
    except:
        return "AI suggestions currently unavailable"

# Main Application
def main():
    st.set_page_config(page_title="CMU Engineering Advisor", layout="wide")
    st.title("🎓 CMU Graduate Program Finder")
    st.write("Discover programs matching your academic interests")
    
    # Load and process data
    with st.spinner("Loading latest program information..."):
        df = get_program_data()
        if not df.empty:
            df['embedding'] = df.apply(
                lambda x: get_ai_embedding(f"{x['name']} {x['description']} {x['courses']}"), 
                axis=1
            )
    
    # Search interface
    search_query = st.text_input("Describe your academic interests:", 
                               placeholder="e.g., 'Robotics and machine learning in healthcare'")
    
    if search_query and not df.empty:
        with st.spinner("Analyzing programs..."):
            query_embedding = get_ai_embedding(search_query)
            if query_embedding:
                df['match_score'] = df['embedding'].apply(
                    lambda x: cosine_similarity([query_embedding], [x])[0][0] if x else 0
                )
                results = df.sort_values('match_score', ascending=False).head(3)
                
                # Display results
                for _, program in results.iterrows():
                    with st.expander(f"🌟 {program['name']} ({program['match_score']:.0%} Match)"):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Description:** {program['description'][:500]}...")
                            if program['courses']:
                                st.markdown(f"**Key Courses:**\n{program['courses'][:300]}...")
                            st.markdown(f"**Admission Requirements:**\n{program['admission'][:600]}...")
                        with col2:
                            st.link_button("Program Website", program['url'])
                            st.markdown(f"**Contact:**\n{program['contact'].split('\n')[0]}")
                            st.markdown(f"**AI Preparation Tips:**\n{get_ai_advice(program['name'], search_query)}")
            else:
                st.error("Couldn't process your query. Please try different keywords.")

if __name__ == "__main__":
    main()
