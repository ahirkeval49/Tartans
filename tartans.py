import streamlit as st
import requests
import numpy as np
import pandas as pd
from requests_html import HTMLSession # The new, powerful library
import time
from urllib.parse import urljoin

# Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_EMBEDDING_URL = "https://api.deepseek.com/v1/embeddings"
CACHE_EXPIRATION = 86400 # Cache data for 24 hours to keep it fresh but fast

# --- NEW: Live scraper using requests-html to handle JavaScript ---
@st.cache_data(ttl=CACHE_EXPIRATION)
def get_live_program_data():
    """
    Scrapes the CMU website live using requests-html to handle dynamic JavaScript content.
    The results are cached for performance and to avoid re-scraping on every interaction.
    """
    base_url = "https://engineering.cmu.edu"
    source_url = f"{base_url}/education/graduate-studies/programs/index.html"
    session = HTMLSession()
    programs = []
    
    st.info("Fetching live data from CMU Engineering... This may take a moment on the first run.")
    
    try:
        r = session.get(source_url, timeout=20)
        
        # This is the crucial step: render the JavaScript on the page.
        # It will wait up to 30 seconds for the content to appear.
        r.html.render(sleep=5, timeout=30)
        
        program_links = r.html.find('div.program-listing h3 a')
        
        if not program_links:
            st.error("Scraper Error: Could not find program links. The website's design may have changed.")
            return pd.DataFrame()

        st.info(f"Found {len(program_links)} programs. Scraping details...")

        for link in program_links:
            program_name = link.text.strip()
            # The link might be relative, so we join it with the base URL
            program_url = urljoin(base_url, list(link.absolute_links)[0])
            
            # Use a new session for each sub-page to be safe
            sub_session = HTMLSession()
            sub_r = sub_session.get(program_url)
            
            description_tag = sub_r.html.find('div.program-intro', first=True)
            description = description_tag.text.strip() if description_tag else 'No detailed description was found on the page.'
            
            programs.append({
                'name': program_name,
                'url': program_url,
                'description': description,
                'department': 'Engineering'
            })
            time.sleep(0.5) # Be polite to the server

    except Exception as e:
        st.error(f"A critical error occurred during scraping: {e}")
        return pd.DataFrame()

    if not programs:
        st.error("Scraping finished, but no program data was collected.")
        return pd.DataFrame()
        
    st.success(f"Successfully scraped and processed {len(programs)} programs.")
    return pd.DataFrame(programs)

# --- AI Functions (no changes needed here) ---
@st.cache_data
def get_ai_embedding(text):
    if 'DEEPSEEK_KEY' not in st.secrets or not st.secrets.DEEPSEEK_KEY:
        st.error("DeepSeek API Key is missing from secrets.")
        return None
    try:
        response = requests.post(DEEPSEEK_EMBEDDING_URL, headers={"Authorization": f"Bearer {st.secrets.DEEPSEEK_KEY}"}, json={"input": text, "model": "deepseek-embedding"}, timeout=20)
        response.raise_for_status()
        return np.array(response.json()['data'][0]['embedding'])
    except Exception as e:
        st.error(f"Embedding API call failed: {e}")
        return None

@st.cache_data
def get_ai_analysis(program_name, program_description, query):
    if 'DEEPSEEK_KEY' not in st.secrets or not st.secrets.DEEPSEEK_KEY: return "AI analysis unavailable: API key missing."
    prompt = f"""As an expert academic advisor, analyze the following program for a student interested in '{query}'.
    Program: {program_name} - {program_description}
    Provide a concise, 3-point analysis:
    - **Profile Match:** Why is this program a strong match?
    - **Key Skills to Highlight:** What skills should the student emphasize in their application?
    - **Potential Career Path:** Mention a specific job title or industry."""
    try:
        response = requests.post(DEEPSEEK_API_URL, headers={"Authorization": f"Bearer {st.secrets.DEEPSEEK_KEY}"}, json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.4, "max_tokens": 350}, timeout=25)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e: return f"AI analysis failed: {e}"

def main():
    st.set_page_config(page_title="CMU Engineering Program Matchmaker", layout="wide")
    st.title("🔍 CMU College of Engineering Program Matchmaker")
    st.write("Discover your ideal engineering graduate program with live, automatically updated data.")
    
    # This single function call now handles the live scraping and caching
    df = get_live_program_data()
    
    if 'DEEPSEEK_KEY' not in st.secrets or not st.secrets.DEEPSEEK_KEY:
        st.error("DeepSeek API Key is NOT found in secrets. Please add it to run the app.")
        return
        
    if df.empty:
        st.warning("Program data could not be loaded from the CMU website. The app cannot continue.")
        return

    # Use session state to avoid re-calculating embeddings on every interaction within a session
    if 'embeddings_generated' not in st.session_state:
        with st.spinner("🧠 Preparing AI embeddings for programs (this happens once per session)..."):
            df['embedding'] = df.apply(lambda row: get_ai_embedding(f"Program: {row['name']}\nDescription: {row['description']}"), axis=1)
            df.dropna(subset=['embedding'], inplace=True)
            st.session_state.program_data = df
            st.session_state.embeddings_generated = True
    
    df = st.session_state.program_data

    search_query = st.text_input("Describe your academic and career interests:", placeholder="e.g., 'robotics and automation in manufacturing'")
    
    if search_query:
        with st.spinner(f"🔍 Finding matches for '{search_query}'..."):
            query_embedding = get_ai_embedding(search_query)
            if query_embedding is not None:
                results_df = df.copy()
                results_df['match_score'] = results_df['embedding'].apply(lambda x: np.dot(query_embedding, x))
                results = results_df.sort_values('match_score', ascending=False).head(3)

                st.subheader(f"Top {len(results)} Program Matches")
                if results.empty: st.warning("No strong matches found.")
                for _, program in results.iterrows():
                    st.markdown(f"#### {program['name']}")
                    st.progress(program['match_score'], text=f"**Match Score: {program['match_score']:.0%}**")
                    with st.container(border=True):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown("**Program Overview**")
                            st.write(program['description'])
                        with col2:
                            st.link_button("🌐 Go to Program Website", program['url'], use_container_width=True)
                        st.markdown("**🤖 AI-Powered Advisor**")
                        analysis = get_ai_analysis(program['name'], program['description'], search_query)
                        st.markdown(analysis)
            else: st.error("Could not process your query.")
    
    st.markdown("---")
    st.markdown("ℹ️ Data is scraped live from the CMU Engineering website and cached for 24 hours. Recommendations are powered by DeepSeek AI.")

if __name__ == "__main__":
    main()
