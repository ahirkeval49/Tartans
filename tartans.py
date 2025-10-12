# --- DATA SCRAPING & CACHING ---
@st.cache_data(ttl=86400) # Cache data for 24 hours
def get_cmu_program_data():
    """
    Scrapes a PREDEFINED LIST of CMU Engineering pages.
    This version uses an UPDATED CSS SELECTOR and includes debugging output if it fails.
    """
    base_url = "https://engineering.cmu.edu"
    source_urls = [
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
        "https://engineering.cmu.edu/education/graduate-studies/programs/cmu-africa.html",
        "https://engineering.cmu.edu/education/graduate-studies/programs/sv.html"
    ]
    critical_url = "https://engineering.cmu.edu/education/graduate-studies/programs/index.html"
    all_programs = {}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    progress_bar = st.progress(0, text="Initializing live data fetch...")
    
    for i, page_url in enumerate(source_urls):
        progress_bar.progress((i + 1) / len(source_urls), text=f"Scanning page: {page_url.split('/')[-1]}")
        try:
            response = requests.get(page_url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # --- CRITICAL FIX [October 2025] ---
            # The old selector 'div.component-content h4 a' is no longer reliable.
            # This new selector is more specific to the container holding program links.
            program_elements = soup.select('div.text-media-right h4 a')
            
            # --- DEBUGGING STEP ---
            # If no elements are found on the main page, show what the scraper sees.
            if not program_elements and page_url == critical_url:
                st.warning("DEBUG: The primary scraper found no program links. The HTML content received by the script was:")
                st.code(response.text[:1000]) # Display the first 1000 characters of the HTML

            for link in program_elements:
                program_name = link.text.strip()
                program_url = urljoin(base_url, link['href'])
                
                if "department" in program_name.lower() or program_url == f"{base_url}/": continue
                
                degree_type = 'Other'
                if 'M.S.' in program_name or 'Master' in program_name: degree_type = 'M.S.'
                elif 'Ph.D.' in program_name: degree_type = 'Ph.D.'
                
                if program_url not in all_programs:
                    all_programs[program_url] = {'name': program_name, 'url': program_url, 'description': '', 'degree_type': degree_type}
        except requests.RequestException as e:
            if page_url == critical_url: st.error(f"Critical source failed: Could not fetch main program page. Error: {e}")
            else: st.warning(f"Could not fetch or process page {page_url.split('/')[-1]}: {e}")
            continue
    
    programs_list = list(all_programs.values())
    total_programs = len(programs_list)
    for i, program in enumerate(programs_list):
        progress_bar.progress((i + 1) / total_programs, text=f"Fetching details for: {program['name']}")
        try:
            sub_response = requests.get(program['url'], headers=headers, timeout=15)
            sub_soup = BeautifulSoup(sub_response.text, 'html.parser')
            description_tag = sub_soup.select_one('div.component-content p')
            program['description'] = description_tag.get_text(strip=True) if description_tag else 'No detailed description found.'
            time.sleep(0.1)
        except requests.RequestException:
            program['description'] = 'Could not retrieve description.'
            
    progress_bar.empty()
    if not programs_list:
        st.error("Scraping finished, but no program data was collected. All sources may be down or the website structure has changed.")
        return pd.DataFrame()
        
    st.success(f"Successfully scraped and processed {len(programs_list)} unique graduate programs.")
    return pd.DataFrame(programs_list)
