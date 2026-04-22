import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import os
import time
from datetime import datetime, timedelta
import re
from openai import OpenAI

# --- CONFIGURATION ---
ST_SECRETS_KEY = "DEEPSEEK_API_KEY"
DATA_FILE = "cmu_programs_data.json"
CACHE_DURATION_HOURS = 24
MODEL_NAME = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

st.set_page_config(
    page_title="CMU Engineering Advisor",
    page_icon="🎓",
    layout="centered"
)

# --- SCAPING ENGINE ---
class CMUScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.urls = [
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
            "https://engineering.cmu.edu/education/graduate-studies/programs/ms-aie.html",
            "https://engineering.cmu.edu/education/graduate-studies/programs/cmu-africa.html",
            "https://engineering.cmu.edu/education/graduate-studies/programs/sv.html"
        ]

    def _clean_text(self, text):
        if not text: return ""
        text = text.replace('\xa0', ' ')
        return " ".join(text.split())

    def _extract_content(self, soup):
        content = soup.find('main') or soup.find(id='content') or soup.find(class_='content')
        if content:
            for garbage in content.find_all(['nav', 'script', 'style', 'aside', 'footer']):
                garbage.decompose()
            return self._clean_text(content.get_text(" ", strip=True))
        return ""

    def scrape(self):
        """Scrapes silently in the background."""
        programs = []
        # Removed st.progress to keep it hidden from frontend users
        for url in self.urls:
            try:
                resp = requests.get(url, headers=self.headers, timeout=10)
                if resp.status_code != 200: continue
                soup = BeautifulSoup(resp.content, 'html.parser')
                
                dept_name = self._clean_text(soup.find('h1').text) if soup.find('h1') else "General Program Info"
                content = self._extract_content(soup)
                
                programs.append({
                    "department": dept_name,
                    "url": url,
                    "content": content
                })
                time.sleep(0.5) 
            except Exception:
                pass # Silently fail on individual pages to keep app running
        return programs

# --- DATA MANAGER ---
def get_program_data():
    scraper = CMUScraper()
    
    # Check cache
    if os.path.exists(DATA_FILE):
        try:
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(DATA_FILE))
            if datetime.now() - file_mod_time < timedelta(hours=CACHE_DURATION_HOURS):
                with open(DATA_FILE, 'r') as f:
                    return json.load(f)
        except:
            pass 

    # If cache miss, scrape (silently with generic spinner)
    with st.spinner("Initializing Advisor Database..."):
        data = scraper.scrape()
        
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)
        
    return data

# --- RETRIEVAL ENGINE (The fix for 429 Errors) ---
def retrieve_relevant_chunks(query, data, top_k=4):
    """
    Instead of sending ALL data, we score each department's content 
    based on how many times the user's keywords appear.
    """
    query_tokens = set(re.findall(r'\w+', query.lower()))
    
    scores = []
    for item in data:
        score = 0
        text_lower = item['content'].lower()
        dept_lower = item['department'].lower()
        
        # 1. Exact Department Match gets huge boost
        if dept_lower in query.lower():
            score += 100
            
        # 2. Keyword frequency
        for token in query_tokens:
            if len(token) > 3: # Ignore small words like 'the', 'how'
                count = text_lower.count(token)
                score += count
        
        scores.append((score, item))
    
    # Sort by relevance and take top K
    scores.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scores[:top_k]]

# --- DEEPSEEK AI ENGINE ---
def generate_response(user_input, chat_history, all_data):
    api_key = st.secrets.get(ST_SECRETS_KEY)
    if not api_key:
        return "Error: API Key missing."

    # Initialize DeepSeek client
    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL
    )
    
    # 1. RETRIEVE only relevant data (Fixes 429 Error)
    relevant_data = retrieve_relevant_chunks(user_input, all_data)
    
    # 2. Build Context String
    context_str = ""
    for item in relevant_data:
        context_str += f"\n[SOURCE: {item['department']} | URL: {item['url']}]\n{item['content'][:8000]}\n" # Cap chunk size just in case

    # 3. System Prompt
    system_prompt = f"""
You are the CMU Engineering Advisor. Answer the user's question using ONLY the context provided below.

- If the answer is not in the context, say "I don't have that specific information in my database."
- Be professional, concise, and helpful.
- Always cite the specific Department or URL when providing details.

CONTEXT:
{context_str}
"""

    # 4. Build conversation messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history (excluding the current user message)
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current user input
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Service is busy. Please try again in a moment. (Error: {str(e)})"

# --- MAIN UI ---
def main():
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/b/bb/Carnegie_Mellon_University_seal.svg/1200px-Carnegie_Mellon_University_seal.svg.png", width=70)
    with col2:
        st.title("CMU Engineering Advisor")
        # Caption removed as requested

    program_data = get_program_data()
    
    if not program_data:
        st.error("Unable to load data.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can help you explore graduate programs at CMU Engineering. What are your research interests?"}
        ]

    for msg in st.session_state.messages:
        avatar = "🎓" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about programs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Thinking..."):
                response_text = generate_response(
                    prompt, 
                    st.session_state.messages[:-1], 
                    program_data
                )
                st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()
