import streamlit as st
import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urljoin

# Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_EMBEDDING_URL = "https://api.deepseek.com/v1/embeddings"
CACHE_EXPIRATION = 86400  # 24 hours in seconds

@st.cache_data(ttl=CACHE_EXPIRATION)
def get_program_data():
    try:
        base_url = "https://engineering.cmu.edu"
        source_url = "https://engineering.cmu.edu/education/graduate-studies/programs/index.html"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(source_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        programs = []
        
        # Updated selector for 2024 website
        program_cards = soup.select('div.accordion-item.program-listing')
        
        for card in program_cards:
            try:
                program_name = card.select_one('h3').text.strip()
                program_url = urljoin(base_url, card.select_one('a.button')['href'])
                
                # Get program details
                program_response = requests.get(program_url, headers=headers, timeout=15)
                program_soup = BeautifulSoup(program_response.text, 'html.parser')
                
                # Extract description
                description = program_soup.find('div', class_='program-intro').get_text(' ', strip=True) if program_soup.find('div', class_='program-intro') else ''
                
                # Extract curriculum
                courses = []
                curriculum_section = program_soup.find('h2', string=lambda t: t and 'curriculum' in t.lower())
                if curriculum_section:
                    for item in curriculum_section.find_next('div').select('li'):
                        courses.append(item.get_text(' ', strip=True))
                
                programs.append({
                    'name': program_name,
                    'url': program_url,
                    'description': description,
                    'courses': '\n'.join(courses),
                    'department': 'Engineering'  # Update based on actual data
                })
                
            except Exception as e:
                print(f"Error processing {program_name}: {str(e)}")
                continue
        
        return pd.DataFrame(programs)
    
    except Exception as e:
        print(f"Scraping failed: {str(e)}")
        return pd.DataFrame()

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
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return None

@st.cache_data
def get_ai_analysis(program_name, query, context):
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
        return response.json()['choices'][0]['message']['content']
    except:
        return "AI analysis currently unavailable"

def main():
    st.set_page_config(page_title="CMU Engineering Advisor Pro", layout="wide")
    st.title("🔍 CMU Graduate Program Matchmaker")
    st.write("Discover your ideal engineering graduate program using AI-powered matching")
    
    # Load data
    with st.spinner("🔄 Loading latest program information from CMU..."):
        df = get_program_data()
        if not df.empty:
            df['embedding'] = df.apply(
                lambda x: get_ai_embedding(f"Program: {x['name']}\nDescription: {x['description']}\nCourses: {x['courses']}"), 
                axis=1
            )
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Describe your academic/career interests:", 
                                   placeholder="e.g., 'Applying machine learning to sustainable energy systems'")
    with col2:
        dept_filter = st.multiselect("Filter by department:", options=df['department'].unique() if not df.empty else [])
    
    if search_query and not df.empty:
        with st.spinner("🔍 Finding best matches..."):
            query_embedding = get_ai_embedding(search_query)
            if query_embedding:
                # Filter dataframe
                filtered_df = df[df['department'].isin(dept_filter)] if dept_filter else df
                
                # Calculate similarity
                filtered_df['match_score'] = filtered_df['embedding'].apply(
                    lambda x: cosine_similarity([query_embedding], [x])[0][0] if x else 0
                )
                results = filtered_df.sort_values('match_score', ascending=False).head(5)
                
                # Display results
                st.subheader(f"Top {len(results)} Program Matches")
                for idx, (_, program) in enumerate(results.iterrows(), 1):
                    with st.expander(f"{idx}. 🥇 {program['name']} ({program['match_score']:.0%} Match)", expanded=idx==1):
                        col_a, col_b = st.columns([3, 1])
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
                            context = f"Description: {program['description'][:500]}\nCourses: {program['courses'][:300]}"
                            analysis = get_ai_analysis(program['name'], search_query, context)
                            st.markdown(f"**🤖 AI Program Analysis:**\n{analysis}")
            else:
                st.error("Error processing your query. Please try rephrasing.")
    
    # Add footer
    st.markdown("---")
    st.markdown("ℹ️ Data sourced from CMU Engineering website. Recommendations powered by DeepSeek AI.")

if __name__ == "__main__":
    main()
