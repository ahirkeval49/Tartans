import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta

# --- CONFIGURATION ---
ST_SECRETS_KEY = "GEMINI_API_KEY"
DATA_FILE = "cmu_programs_data.json"
CACHE_DURATION_HOURS = 24
MODEL_NAME = "gemini-2.0-flash" 

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
        # CMU Department Graduate Program Landing Pages
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
        """Standardizes text: removes extra spaces, newlines, and non-ascii junk."""
        if not text: return ""
        text = text.replace('\xa0', ' ')
        return " ".join(text.split())

    def _extract_content(self, soup):
        """Targeted extraction for CMU Engineering templates."""
        # CMU usually puts main content in 'main' role or specific IDs
        content = soup.find('main') or soup.find(id='content') or soup.find(class_='content')
        
        if content:
            # Remove navs, sidebars, scripts within the main content if they exist
            for garbage in content.find_all(['nav', 'script', 'style', 'aside', 'footer']):
                garbage.decompose()
            return self._clean_text(content.get_text(" ", strip=True))
        return ""

    def scrape(self):
        """Main scraping loop."""
        programs = []
        progress_bar = st.progress(0, text="Updating database...")
        
        for idx, url in enumerate(self.urls):
            try:
                # Update UI
                progress_bar.progress((idx + 1) / len(self.urls), text=f"Scraping {url.split('/')[-1]}...")
                
                resp = requests.get(url, headers=self.headers, timeout=10)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.content, 'html.parser')
                
                # Metadata
                title = soup.title.string if soup.title else "CMU Program"
                dept_name = self._clean_text(soup.find('h1').text) if soup.find('h1') else "Unknown Department"
                content = self._extract_content(soup)
                
                # Logic: One page might contain info for both MS and PhD.
                # We will store the page as a knowledge unit.
                programs.append({
                    "department": dept_name,
                    "url": url,
                    "content": content[:15000] # Cap text length per page to avoid outliers
                })
                
                # Polite delay
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")
        
        progress_bar.empty()
        return programs

# --- DATA MANAGER ---
def get_program_data():
    """
    Checks for local JSON. 
    If exists and fresh (<24h) -> Load it.
    Else -> Scrape, Save, Load.
    """
    scraper = CMUScraper()
    
    # Check if file exists
    if os.path.exists(DATA_FILE):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(DATA_FILE))
        if datetime.now() - file_mod_time < timedelta(hours=CACHE_DURATION_HOURS):
            try:
                with open(DATA_FILE, 'r') as f:
                    data = json.load(f)
                return data
            except:
                pass # If file is corrupted, fall through to re-scrape

    # If we are here, we need to scrape
    with st.spinner("Refreshing CMU Program Database (runs once per day)..."):
        data = scraper.scrape()
        
    # Save to file
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)
        
    return data

# --- GEMINI AI ENGINE ---
def generate_response(user_input, chat_history, context_data):
    """
    Constructs the prompt with the FULL context and sends to Gemini.
    """
    api_key = st.secrets.get(ST_SECRETS_KEY)
    if not api_key:
        return "Error: Gemini API Key not found in secrets."

    genai.configure(api_key=api_key)
    
    # 1. Format the context from our data
    context_str = ""
    for item in context_data:
        context_str += f"\n--- SOURCE START: {item['department']} ---\n"
        context_str += f"URL: {item['url']}\n"
        context_str += f"CONTENT: {item['content']}\n"
        context_str += "--- SOURCE END ---\n"

    # 2. System Prompt with Visual Instructions
    system_instruction = f"""
    You are the 'CMU Engineering Advisor', a helpful assistant for prospective graduate students.
    
    YOUR KNOWLEDGE BASE:
    I have provided the full text content of the CMU College of Engineering graduate program pages below. 
    Answer the user's questions based ONLY on this provided context. If the answer isn't in the context, say you don't know.
    
    BEHAVIOR:
    - Be enthusiastic and welcoming.
    - If the user asks for program recommendations, analyze the provided context to find the best matches based on their interests.
    - Always provide the specific URL when mentioning a program.
    - Analyze the user's intent. If they are asking about specific research areas (e.g., "AI", "Robotics", "Sustainability"), cross-reference all departments.
    
    VISUAL AIDS:
    - Assess if the user would understand the response better with a diagram.
    - Insert a diagram tag by adding 

[Image of X]
 where X is a specific query.
    - Examples: , , 

[Image of neural network architecture]
.
    - Only use tags if they add instructional value.
    
    CONTEXT DATA:
    {context_str}
    """

    # 3. Build Model
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_instruction
    )

    # 4. Convert Streamlit history to Gemini history
    gemini_history = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    # 5. Generate
    try:
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(user_input)
        return response.text
    except Exception as e:
        return f"I encountered an error connecting to Gemini: {str(e)}"

# --- MAIN UI ---
def main():
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/b/bb/Carnegie_Mellon_University_seal.svg/1200px-Carnegie_Mellon_University_seal.svg.png", width=70)
    with col2:
        st.title("CMU Engineering Advisor")
        st.caption("Powered by Gemini 1.5 • Daily Synchronized Data")

    # Load Data (Instant if cached, Slower if < 24h old)
    program_data = get_program_data()
    
    if not program_data:
        st.error("Could not retrieve program data. Please check your internet connection.")
        st.stop()

    # Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can help you navigate the Graduate Programs at CMU Engineering. Are you interested in a Master's or Ph.D., or do you have a specific research interest in mind?"}
        ]

    # Display Chat
    for msg in st.session_state.messages:
        avatar = "🎓" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask about programs, admission requirements, or research..."):
        # User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Thinking..."):
                response_text = generate_response(
                    prompt, 
                    st.session_state.messages[:-1], # Pass history excluding current prompt (Gemini handles the prompt separately in send_message)
                    program_data
                )
                st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()
