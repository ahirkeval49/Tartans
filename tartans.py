import streamlit as st
import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_EMBEDDING_URL = "https://api.deepseek.com/v1/embeddings"
CACHE_EXPIRATION = 86400 # Cache for 24 hours

# --- NEW: Selenium-powered scraper that handles JavaScript ---
@st.cache_data(ttl=CACHE_EXPIRATION)
def get_program_data():
    """
    Scrapes the CMU website using Selenium to handle dynamic JavaScript content.
    """
    base_url = "https://engineering.cmu.edu"
    source_url = f"{base_url}/education/graduate-studies/programs/index.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    # --- Selenium Setup for Streamlit Cloud ---
    chrome_options = Options()
    chrome_options.add_argument("--headless") # Must run in headless mode on the server
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    programs = []
    try:
        driver.get(source_url)
        
        # --- Wait for the program list to load dynamically ---
        # This is the crucial step: we wait up to 15 seconds for the JS to render the program listings.
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.program-listing")))
        
        # Now that the page is loaded, get the HTML and pass it to BeautifulSoup
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        
        program_links = soup.select('div.program-listing h3 a')
        st.info(f"Found {len(program_links)} potential programs. Fetching details...")

        if not program_links:
            st.warning("Scraper ran, but could not find any program links. The website structure may have changed.")
            return pd.DataFrame()

        for link in program_links:
            program_name = link.get_text(strip=True)
            program_url = urljoin(base_url, link.get('href'))
            
            # Use requests for sub-pages as it's faster than using Selenium for each one
            time.sleep(0.25) 
            prog_response = requests.get(program_url, headers=headers, timeout=15)
            prog_soup = BeautifulSoup(prog_response.text, 'html.parser')

            description_tag = prog_soup.find('div', class_='program-intro')
            description = description_tag.get_text(strip=True) if description_tag else 'No detailed description was found on the program page.'

            programs.append({
                'name': program_name, 'url': program_url, 'description': description,
                'department': 'Engineering'
            })
            
    except Exception as e:
        st.error(f"An error occurred during scraping: {e}")
        return pd.DataFrame()
    finally:
        # Important to close the browser to free up resources
        driver.quit()

    if not programs:
        st.error("Scraping finished, but no program data was collected.")
        return pd.DataFrame()
    
    st.success(f"Successfully scraped and processed {len(programs)} programs.")
    return pd.DataFrame(programs)

# --- AI Functions (no changes needed here) ---
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
def get_ai_analysis(program_name, program_description, query):
    if 'DEEPSEEK_KEY' not in st.secrets or not st.secrets.DEEPSEEK_KEY:
        return "AI analysis unavailable: API key missing."
    prompt = f"""
    As an expert academic advisor, analyze the following graduate program for a prospective student.
    Student's Interest: "{query}"
    Program Name: {program_name}
    Program Description: "{program_description}"
    Provide a concise, 3-point analysis in bullet points:
    - **Profile Match:** Why is this program a strong match for the student's interest?
    - **Key Skills to Highlight:** What specific skills or experiences should the student emphasize in their application for this program?
    - **Potential Career Path:** Mention a specific job title or industry this program prepares graduates for.
    """
    try:
        response = requests.post(DEEPSEEK_API_URL, headers={"Authorization": f"Bearer {st.secrets.DEEPSEEK_KEY}"}, json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.4, "max_tokens": 350}, timeout=25)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"AI analysis failed: {e}"

def main():
    st.set_page_config(page_title="CMU Engineering Program Matchmaker", layout="wide")
    st.title("🔍 CMU College of Engineering Program Matchmaker")
    st.write("Discover your ideal engineering graduate program with live data from the CMU website.")
    
    df = get_program_data()
        
    if 'DEEPSEEK_KEY' not in st.secrets or not st.secrets.DEEPSEEK_KEY:
        st.error("DeepSeek API Key is NOT found in secrets. Please add it to run the app.")
        return

    if df.empty:
        st.warning("Program data could not be loaded. The application cannot proceed.")
        return

    # Use session state to avoid re-calculating embeddings on every interaction
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
                if results.empty:
                    st.warning("No strong matches found. Try rephrasing your interests.")

                for _, program in results.iterrows():
                    match_score = program['match_score']
                    st.markdown(f"#### {program['name']}")
                    st.progress(match_score, text=f"**Match Score: {match_score:.0%}**")
                    
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
            else:
                st.error("Could not process your query. The embedding API might be down or your key is invalid.")
    
    st.markdown("---")
    st.markdown("ℹ️ Data is scraped live from the CMU Engineering website. Recommendations are powered by DeepSeek AI.")

if __name__ == "__main__":
    main()
