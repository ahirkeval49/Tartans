import streamlit as st
import requests
import numpy as np
import pandas as pd
import time

# Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_EMBEDDING_URL = "https://api.deepseek.com/v1/embeddings"

@st.cache_data
def load_data_from_csv(file_path="cmu_programs.csv"):
    try:
        df = pd.read_csv(file_path)
        df['department'] = 'Engineering'
        st.success(f"Successfully loaded {len(df)} programs from local data.")
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file '{file_path}' was not found. Please run the scraper first.")
        return pd.DataFrame()

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
    st.write("Discover your ideal engineering graduate program with automatically updated data from the CMU website.")
    
    if 'program_data' not in st.session_state:
        df = load_data_from_csv()
        if not df.empty:
            with st.spinner("🧠 Preparing AI embeddings for programs (this happens once)..."):
                df['embedding'] = df.apply(lambda row: get_ai_embedding(f"Program: {row['name']}\nDescription: {row['description']}"), axis=1)
                df.dropna(subset=['embedding'], inplace=True)
                st.session_state.program_data = df
        else: st.session_state.program_data = pd.DataFrame()
    df = st.session_state.program_data

    if df.empty:
        st.warning("Program data is not available. The application cannot proceed.")
        return

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
    st.markdown("ℹ️ Data is updated weekly via an automated scraper. Recommendations are powered by DeepSeek AI.")

if __name__ == "__main__":
    main()
