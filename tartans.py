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
    base_url = "https://engineering.cmu.edu"
    source_url = f"{base_url}/education/graduate-studies/programs/index.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }

    programs = []
    try:
        response = requests.get(source_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Locate all program links
        program_links = soup.select('div.accordion-item.program-listing a.button')
        for link in program_links:
            program_name = link.get_text(strip=True)
            program_url = urljoin(base_url, link['href'])

            # Fetch program-specific page
            prog_response = requests.get(program_url, headers=headers, timeout=15)
            prog_soup = BeautifulSoup(prog_response.text, 'html.parser')

            # Extract description
            description_tag = prog_soup.find('div', class_='program-intro')
            description = description_tag.get_text(strip=True) if description_tag else ''

            # Extract courses
            courses = []
            curriculum_header = prog_soup.find('h2', string=lambda t: t and 'curriculum' in t.lower())
            if curriculum_header:
                curriculum_div = curriculum_header.find_next_sibling('div')
                if curriculum_div:
                    courses = [li.get_text(strip=True) for li in curriculum_div.find_all('li')]

            # Extract admission requirements
            admission = ''
            admission_header = prog_soup.find('h2', string=lambda t: t and 'admission' in t.lower())
            if admission_header:
                admission_div = admission_header.find_next_sibling('div')
                if admission_div:
                    admission = admission_div.get_text(strip=True)

            # Extract contact information
            contact = ''
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
                'department': 'Engineering'  # Update based on actual data
            })

    except Exception as e:
        print(f"Scraping failed: {e}")

    return pd.DataFrame(programs)


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
