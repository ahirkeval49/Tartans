import pandas as pd
from requests_html import HTMLSession
from urllib.parse import urljoin
import time

def scrape_cmu_programs():
    """
    Scrapes the CMU website for graduate programs and saves them to a CSV file.
    Uses requests-html to handle JavaScript rendering.
    """
    base_url = "https://engineering.cmu.edu"
    source_url = f"{base_url}/education/graduate-studies/programs/index.html"
    session = HTMLSession()
    programs = []

    print("Starting scrape...")
    try:
        r = session.get(source_url)
        # Wait for the JavaScript to render the page content
        r.html.render(sleep=5, timeout=20)
        
        program_links = r.html.find('div.program-listing h3 a')
        print(f"Found {len(program_links)} programs.")

        for link in program_links:
            program_name = link.text.strip()
            # The link might be relative, so we join it with the base URL
            program_url = urljoin(base_url, list(link.absolute_links)[0])
            
            print(f"Scraping details for: {program_name}")
            # Use a new session for each sub-page to be safe
            sub_session = HTMLSession()
            sub_r = sub_session.get(program_url)
            
            description_tag = sub_r.html.find('div.program-intro', first=True)
            description = description_tag.text.strip() if description_tag else 'No detailed description was found.'
            
            programs.append({
                'name': program_name,
                'url': program_url,
                'description': description
            })
            time.sleep(0.5) # Be polite to the server

    except Exception as e:
        print(f"An error occurred during scraping: {e}")
        return

    if not programs:
        print("Scraping finished, but no program data was collected.")
        return
        
    df = pd.DataFrame(programs)
    df.to_csv("cmu_programs.csv", index=False)
    print(f"Successfully scraped {len(df)} programs and saved to cmu_programs.csv")

if __name__ == "__main__":
    scrape_cmu_programs()
