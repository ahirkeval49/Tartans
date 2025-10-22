import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
import time
import re # For finding contact info
from typing import Optional, Dict, Any, List

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CMU Engineering Program Finder",
    page_icon="🎓",
    layout="wide"
)

# --- UTILITIES ---

def robust_request(url: str, headers: Dict[str, str], timeout: int = 15) -> Optional[requests.Response]:
    """Handles HTTP requests with basic error checking, returns the Response object."""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        # Don't show warnings directly in UI during scraping
        # st.warning(f"Request failed for {url}: {e}")
        print(f"Request failed for {url}: {e}") # Log to console instead
        return None

def extract_contact_info(soup: BeautifulSoup) -> str:
    """Attempts to extract email or relevant contact links from the page."""
    # Prioritize mailto links
    mail_links = soup.find_all('a', href=lambda href: href and href.startswith('mailto:'))
    if mail_links:
        # Return the first relevant email, clean it up
        email = mail_links[0]['href'].replace('mailto:', '').split('?')[0]
        # Avoid generic webmaster emails if possible
        if not any(x in email for x in ['webmaster', 'support', 'info@', 'help@']):
             return f"Email: {email}"
        elif len(mail_links) > 1:
             # Try the second link if the first was generic
             email = mail_links[1]['href'].replace('mailto:', '').split('?')[0]
             if not any(x in email for x in ['webmaster', 'support', 'info@', 'help@']):
                  return f"Email: {email}"

    # Look for contact page links within main content
    main_content = soup.find('main') or soup.find('article') or soup.body
    if main_content:
        contact_links = main_content.find_all('a', string=re.compile(r'contact|directory|people', re.IGNORECASE))
        if contact_links:
            contact_url = contact_links[0].get('href', '')
            # Make URL absolute if relative
            if contact_url and not contact_url.startswith(('http:', 'https:', '#', 'mailto:')):
                 # Need base URL - tricky without passing it in, make a guess or skip
                 pass # Skip relative URLs for now to avoid complexity
            elif contact_url.startswith('http'):
                 return f"Contact Page: {contact_url}"

    # Fallback: return the first found mailto link even if generic, or none
    if mail_links:
         email = mail_links[0]['href'].replace('mailto:', '').split('?')[0]
         return f"Email: {email}"

    return "Contact info not readily found on page."


# --- DATA SCRAPING & PROCESSING ---

@st.cache_data(ttl=86400) # Cache data for 24 hours
def get_cmu_program_data():
    """Scrapes CMU program data including name, description, and contact info."""
    department_urls = [
        "https://engineering.cmu.edu/education/graduate-studies/programs/bme.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/cheme.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/cee.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/ece.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/epp.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/ini.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/iii.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/mse.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/meche.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/cmu-africa.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/sv.html"
        "https://engineering.cmu.edu/education/graduate-studies/programs/ms-aie.html"
    ]

    programs_list = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    # Use a hidden progress bar or just log to console
    # progress_bar = st.progress(0, text="Loading program data...")
    print("Starting CMU program data scrape...")

    for i, page_url in enumerate(department_urls):
        page_name = page_url.split('/')[-1]
        # progress_bar.progress((i + 1) / len(department_urls), text=f"Processing: {page_name}")
        print(f"Processing: {page_url}")

        response = robust_request(page_url, headers)
        if not response:
            print(f"Skipping page: {page_name} (Failed to retrieve).")
            continue

        # Check for non-HTML content types which might indicate download links or errors
        content_type = response.headers.get('Content-Type', '')
        if 'html' not in content_type.lower():
             print(f"Skipping page: {page_name} (Content-Type is not HTML: {content_type}).")
             continue

        try:
             soup = BeautifulSoup(response.text, 'html.parser')

             # Remove comments, script, style tags before processing text
             for element in soup(["script", "style", "header", "footer", "nav"]):
                 element.decompose()
             for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                 comment.extract()


             program_name_base = "CMU Program"
             h1 = soup.find('h1')
             title = soup.find('title')
             if h1:
                 program_name_base = h1.get_text(strip=True).replace("Graduate", "").strip()
             elif title:
                 program_name_base = title.get_text(strip=True).split('|')[0].strip()

             description_text = 'No detailed description extracted.'
             main_content = soup.find('main') or soup.find('article') or soup.body
             if main_content:
                 paragraphs = main_content.find_all('p')
                 text_parts = [p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 100] # Stricter length
                 if text_parts:
                     description_text = ' '.join(text_parts)
                     max_desc_len = 1000 # Keep descriptions reasonably concise
                     if len(description_text) > max_desc_len:
                         description_text = description_text[:max_desc_len].rsplit(' ', 1)[0] + '...'
             else:
                  print(f"Could not find main content area for {page_name}")

             # Extract contact info
             contact = extract_contact_info(soup)

             # Add both MS and PhD variants
             programs_list.append({
                 'name': f"Master of Science ({program_name_base})",
                 'url': page_url, 'description': description_text, 'degree_type': 'M.S.', 'contact': contact
             })
             programs_list.append({
                 'name': f"Doctor of Philosophy ({program_name_base})",
                 'url': page_url, 'description': description_text, 'degree_type': 'Ph.D.', 'contact': contact
             })

             print(f"Indexed M.S. and Ph.D. from: {program_name_base}")
             time.sleep(0.1) # Be polite

        except Exception as e:
            print(f"Error parsing {page_url}: {e}")
            continue # Skip to next URL if parsing fails


    # progress_bar.empty() # Remove progress bar

    if not programs_list:
        # Show error in UI if scraping completely fails
        st.error("Scraping finished, but no program data was collected. Please check the source URLs or website structure.")
        return pd.DataFrame()

    print(f"Scraped and indexed {len(programs_list)} program variants.") # Log completion
    return pd.DataFrame(programs_list)


# --- KEYWORD SEARCH FUNCTION ---

def find_matching_programs(keywords_str: str, df_programs: pd.DataFrame, degree_level: str) -> pd.DataFrame:
    """Filters programs based on keywords and degree level."""
    if not keywords_str or df_programs.empty:
        return pd.DataFrame() # Return empty if no keywords or data

    keywords = [kw.strip().lower() for kw in keywords_str.split(',') if kw.strip()]
    if not keywords:
        return pd.DataFrame() # Return empty if keywords are just whitespace/commas

    matches = []

    # Filter by degree level first
    df_filtered_degree = df_programs[df_programs['degree_type'] == degree_level].copy()


    for index, row in df_filtered_degree.iterrows():
        text_to_search = f"{row['name']} {row['description']}".lower()
        score = 0
        matched_keywords = []

        for kw in keywords:
            if kw in text_to_search:
                score += text_to_search.count(kw) # Simple count scoring
                matched_keywords.append(kw)

        if score > 0:
            row_dict = row.to_dict()
            row_dict['score'] = score
            row_dict['matched_keywords'] = matched_keywords
            matches.append(row_dict)

    if not matches:
        return pd.DataFrame()

    # Sort by score descending
    matches_df = pd.DataFrame(matches)
    matches_df.sort_values('score', ascending=False, inplace=True)

    return matches_df


# --- MAIN APPLICATION LOGIC ---
def main():
    st.title("🎓 CMU Engineering Program Finder")
    st.markdown("Enter keywords (separated by commas) to find relevant graduate programs.")

    # --- REMOVED SIDEBAR CONFIGURATION ---

    # Load and process data (runs once due to caching)
    # Show spinner only for the initial data load
    with st.spinner("Loading and indexing program data... (This happens once)"):
        df_programs = get_cmu_program_data()

    if df_programs.empty:
        st.error("Program data could not be loaded. Cannot proceed.")
        st.stop() # Stop execution if data loading fails

    # --- Keyword Input ---
    st.subheader("Search for Programs")
    degree_level = st.radio("Select Degree Level:", ("M.S.", "Ph.D."), horizontal=True, key="degree_level_filter")
    keywords_input = st.text_area(
        "Enter keywords related to your interests (e.g., robotics, machine learning, sustainable energy):",
        placeholder="Separate keywords with commas",
        height=100
    )

    if keywords_input and st.button("Search Programs"):
        # --- Perform Keyword Search ---
        with st.spinner("Searching for matching programs..."):
            results_df = find_matching_programs(keywords_input, df_programs, degree_level)

        # --- Display Direct Results ---
        st.subheader(f"Matching {degree_level} Programs Found:")

        if results_df.empty:
            st.warning("No programs found matching your keywords for the selected degree level. Try different or broader terms.")
        else:
            st.write(f"Found {len(results_df)} matching program(s). Displaying top results:")
            # Display top N results (e.g., top 10)
            for i, (_, program) in enumerate(results_df.head(10).iterrows()):
                st.markdown("---")
                st.markdown(f"### **{i+1}. {program['name']}**")

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Matched Keywords:** {', '.join(program['matched_keywords'])}")
                    st.write(f"**Contact:** {program['contact']}")
                with col2:
                    st.link_button("Go to Program Page ↗️", program['url'], use_container_width=True) # Kept use_container_width here

                with st.expander("Program Description Snippet"):
                    st.write(program['description'])

    else:
        st.info("Enter some keywords and click 'Search Programs' to begin.")


if __name__ == "__main__":
    main()
