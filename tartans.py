# tartans.py
# CMU Engineering Program Conversational Advisor
# Author: Gemini (Google AI)
# Date: October 22, 2025 (Conversational RAG Version - v7.0)

import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
import time
import re # For finding contact info and cleaning
import google.generativeai as genai
from typing import Optional, Dict, Any, List

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CMU Engineering Advisor",
    page_icon="🤖",
    layout="wide"
)

# --- UTILITIES ---

def robust_request(url: str, headers: Dict[str, str], timeout: int = 15) -> Optional[requests.Response]:
    """Handles HTTP requests, returns Response object."""
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return None

def extract_contact_info(soup: BeautifulSoup) -> str:
    """Attempts to extract email or relevant contact links."""
    # Simplified extraction logic
    mail_links = soup.find_all('a', href=lambda href: href and href.startswith('mailto:'))
    if mail_links:
        for link in mail_links:
            email = link['href'].replace('mailto:', '').split('?')[0].strip()
            # Try to find non-generic emails first
            if email and not any(x in email for x in ['webmaster', 'support', 'help@', 'noreply', 'info@']):
                 return f"Email: {email}"
        # Fallback to the first email if no ideal one found
        if mail_links:
            return f"Email: {mail_links[0]['href'].replace('mailto:', '').split('?')[0].strip()}"

    # Look for contact links broadly
    contact_links = soup.find_all('a', string=re.compile(r'contact|directory|people|email|staff|faculty', re.IGNORECASE))
    if contact_links:
         for link in contact_links:
              href = link.get('href', '')
              # Prioritize URLs with 'contact' in the path or text
              link_text_lower = link.get_text().lower()
              if href.startswith('http') and ('contact' in href or 'contact' in link_text_lower):
                   return f"Contact Page: {href}"
         # Fallback to first http link found among these keywords
         for link in contact_links:
              href = link.get('href', '')
              if href.startswith('http'):
                   return f"Possible Contact Link: {href}"

    return "Contact info not readily found."

def clean_text(text):
    """Removes excessive whitespace and short lines."""
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Keep lines with more than 3 words to filter out isolated labels/links
    cleaned_text = '\n'.join(chunk for chunk in chunks if len(chunk.split()) > 3)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text) # Max 2 consecutive newlines
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text) # Replace multiple spaces with single
    return cleaned_text.strip()

# --- DATA SCRAPING & PROCESSING ---

@st.cache_data(ttl=86400) # Cache data for 24 hours
def get_cmu_program_data():
    """Scrapes CMU program data and returns a DataFrame."""
    department_urls = [
        "https://engineering.cmu.edu/education/graduate-studies/programs/index.html", # Main Grad Programs Page
        "https://engineering.cmu.edu/education/graduate-studies/programs/bme.html", # Biomedical Engineering
        "https://engineering.cmu.edu/education/graduate-studies/programs/cheme.html", # Chemical Engineering
        "https://engineering.cmu.edu/education/graduate-studies/programs/cee.html", # Civil & Environmental Engineering
        "https://engineering.cmu.edu/education/graduate-studies/programs/ece.html", # Electrical & Computer Engineering
        "https://engineering.cmu.edu/education/graduate-studies/programs/epp.html", # Engineering & Public Policy
        "https://engineering.cmu.edu/education/graduate-studies/programs/ini.html", # Information Networking Institute
        "https://engineering.cmu.edu/education/graduate-studies/programs/iii.html", # Integrated Innovation Institute
        "https://engineering.cmu.edu/education/graduate-studies/programs/mse.html", # Materials Science & Engineering
        "https://engineering.cmu.edu/education/graduate-studies/programs/meche.html", # Mechanical Engineering
        "https://engineering.cmu.edu/education/graduate-studies/programs/ms-aie.html", # M.S. Artificial Intelligence Engineering
        "https://engineering.cmu.edu/education/graduate-studies/programs/cmu-africa.html", # CMU Africa
        "https://engineering.cmu.edu/education/graduate-studies/programs/sv.html" # Silicon Valley
    ]

    programs_list = []
    headers = {'User-Agent': 'CMUProgramFinderBot/1.0 (Python requests; +streamlit.app)'}
    print("Starting CMU program data scrape...")

    for i, page_url in enumerate(department_urls):
        page_name = page_url.split('/')[-1]
        print(f"Processing ({i+1}/{len(department_urls)}): {page_url}")
        response = robust_request(page_url, headers)
        if not response or 'html' not in response.headers.get('Content-Type', '').lower():
            print(f"Skipping: {page_name} (Failed or not HTML)")
            continue
        try:
            # Use lxml for potentially faster parsing if installed, fallback to html.parser
            try:
                 soup = BeautifulSoup(response.text, 'lxml')
            except:
                 soup = BeautifulSoup(response.text, 'html.parser')

            # --- Remove common noise elements ---
            selectors_to_remove = ["script", "style", "header", "footer", "nav", "aside",
                                  ".sidebar", "#sidebar", ".related-links", ".breadcrumb",
                                  "form", "button", "iframe", ".skip-link", ".screen-reader-text"]
            for selector in selectors_to_remove:
                 elements = soup.select(selector) # Use select for CSS selectors
                 for element in elements:
                      element.decompose()
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

            # --- Extract Program Name ---
            program_name_base = "CMU Program"
            h1 = soup.find('h1')
            title = soup.find('title')
            # Prefer H1 if it seems specific, otherwise fallback to title
            if h1 and len(h1.get_text(strip=True)) > 5 and 'engineering' in h1.get_text(strip=True).lower():
                program_name_base = h1.get_text(strip=True).replace("Graduate", "").strip()
            elif title:
                program_name_base = title.get_text(strip=True).split('|')[0].strip()
            # Refine name if it's too generic like 'Graduate Programs'
            if program_name_base.lower() in ['graduate programs', 'graduate studies']:
                 # Try finding a more specific heading nearby or use URL part
                 h2 = soup.find('h2')
                 if h2: program_name_base = h2.get_text(strip=True)
                 else: program_name_base = page_name.replace('.html','').replace('-',' ').title()


            # --- Find main content area more reliably ---
            content_area = (
                soup.find('main') or # Standard semantic tag
                soup.find(role='main') or # ARIA role
                soup.find(id='main-content') or # Common IDs
                soup.find(id='content') or
                soup.find(class_='main-content') or # Common classes
                soup.find(class_='content') or
                soup.find('article') or # Standard semantic tag
                soup.body # Fallback
            )

            description_text = 'No detailed description extracted.'
            if content_area:
                raw_text = content_area.get_text(separator='\n', strip=True)
                cleaned_description = clean_text(raw_text)

                # Check length *after* cleaning
                if len(cleaned_description) > 150: # Increased minimum length
                    description_text = cleaned_description
                    max_desc_len = 2500 # Increase max length slightly
                    if len(description_text) > max_desc_len:
                        end_pos = description_text.rfind('.', 0, max_desc_len)
                        end_pos = end_pos if end_pos > max_desc_len * 0.7 else description_text.rfind(' ', 0, max_desc_len)
                        description_text = description_text[:(end_pos+1 if end_pos != -1 else max_desc_len)] + '...' # Add 1 to include period
                else:
                    print(f"  -> Cleaned text too short for {page_name} (Length: {len(cleaned_description)})")
            else:
                  print(f"  -> Could not find a suitable content area for {page_name}")

            # --- Extract contact info ---
            contact = extract_contact_info(soup)

            # --- Intelligent Degree Level Detection ---
            page_text_lower = soup.get_text().lower()
            mentions_ms = any(x in page_text_lower for x in ['master of science', ' m.s.', ' master\'s', ' ms '])
            mentions_phd = any(x in page_text_lower for x in ['doctor of philosophy', ' ph.d.', ' doctoral ', ' phd '])

            added_program = False
            # If ONLY MS is clearly mentioned, or NEITHER is mentioned, add MS.
            if (mentions_ms and not mentions_phd) or (not mentions_ms and not mentions_phd):
                programs_list.append({'name': f"Master of Science ({program_name_base})", 'url': page_url, 'description': description_text, 'degree_type': 'M.S.', 'contact': contact})
                print(f"  -> Added M.S. variant for: {program_name_base}")
                added_program = True
            # If ONLY PhD is clearly mentioned, or NEITHER is mentioned, add PhD.
            if (mentions_phd and not mentions_ms) or (not mentions_ms and not mentions_phd):
                programs_list.append({'name': f"Doctor of Philosophy ({program_name_base})", 'url': page_url, 'description': description_text, 'degree_type': 'Ph.D.', 'contact': contact})
                print(f"  -> Added Ph.D. variant for: {program_name_base}")
                added_program = True
            # If BOTH MS and PhD seem to be mentioned, add both.
            if mentions_ms and mentions_phd:
                 if not added_program: # Avoid duplicates if 'neither' case already added them
                    programs_list.append({'name': f"Master of Science ({program_name_base})", 'url': page_url, 'description': description_text, 'degree_type': 'M.S.', 'contact': contact})
                    print(f"  -> Added M.S. variant (both mentioned) for: {program_name_base}")
                    programs_list.append({'name': f"Doctor of Philosophy ({program_name_base})", 'url': page_url, 'description': description_text, 'degree_type': 'Ph.D.', 'contact': contact})
                    print(f"  -> Added Ph.D. variant (both mentioned) for: {program_name_base}")
                    added_program = True

            if not added_program:
                 print(f"  -> Degree level logic check failed for: {program_name_base}. MS:{mentions_ms}, PhD:{mentions_phd}")


            time.sleep(0.05) # Shorter sleep
        except Exception as e:
            print(f"Error parsing {page_url}: {e}")
            continue # Skip to next URL if parsing fails

    if not programs_list:
        # Show error in UI if scraping completely fails
        st.error("Scraping failed: No program data could be collected. Please check the source URLs or website structure.")
        return pd.DataFrame()

    print(f"Scraped and indexed {len(programs_list)} program variants.") # Log completion
    return pd.DataFrame(programs_list)


# --- AI PROVIDER (CHAT) FUNCTIONS ---

def _call_deepseek_chat_api(api_key: str, messages: List[Dict[str, str]], temp: float = 0.5, max_tokens: int = 1500) -> str:
    """Calls DeepSeek chat API."""
    endpoint = "https://api.deepseek.com/v1/chat/completions"
    payload = { "model": "deepseek-chat", "messages": messages, "temperature": temp, "max_tokens": max_tokens, "stream": False }
    for attempt in range(2): # Reduce retries
        try:
            response = requests.post(endpoint, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=90) # Long timeout
            response.raise_for_status()
            json_response = response.json()
            # More robust check for content
            if (choices := json_response.get('choices')) and \
               isinstance(choices, list) and len(choices) > 0 and \
               (message := choices[0].get('message')) and \
               isinstance(message, dict) and \
               (content := message.get('content')):
                return content
            else:
                 error_detail = json_response.get('error', {}).get('message', 'Unexpected format')
                 print(f"DeepSeek API Error: {error_detail} Full Response: {json_response}")
                 return f"AI Error: Unexpected response format ({error_detail})"
        except requests.exceptions.RequestException as e:
            error_msg = f"DeepSeek API call failed: Status {e.response.status_code if e.response else 'N/A'} - {e}"
            if attempt < 1:
                print(f"{error_msg}. Retrying...")
                time.sleep(1.5 ** attempt)
                continue
            print(error_msg)
            return f"AI Error: API call failed ({e})"
        except Exception as e:
            print(f"DeepSeek processing failed: {e}")
            return f"AI Error: Processing failed ({e})"
    return "AI Error: Failed after multiple retries."

def _call_gemini_chat_api(api_key: str, messages: List[Dict[str, str]], temp: float = 0.5, max_tokens: int = 1500) -> str:
    """Calls Gemini chat API, converting message format."""
    try:
        genai.configure(api_key=api_key)
        generation_config = genai.types.GenerationConfig(temperature=temp, max_output_tokens=max_tokens)
        # Use a recent, stable model
        model = genai.GenerativeModel('gemini-1.5-flash-latest', generation_config=generation_config)

        # Convert OpenAI format messages to Gemini format
        gemini_history = []
        system_prompt = None # Gemini prefers system instructions via specific parameter

        # Separate system prompt and build history
        user_model_msgs = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                system_prompt = content # Capture the first system prompt
            elif role in ["user", "assistant"]:
                 user_model_msgs.append(msg) # Keep user/assistant messages for history

        # Build Gemini history, ensuring alternating roles
        last_role = None
        for i, msg in enumerate(user_model_msgs):
             role = msg.get("role")
             content = msg.get("content")
             gemini_role = "user" if role == "user" else "model"

             # Skip empty messages or consecutive messages from the same role
             if not content or gemini_role == last_role:
                  print(f"Skipping invalid/consecutive message: {msg}")
                  continue

             gemini_history.append({'role': gemini_role, 'parts': [content]})
             last_role = gemini_role

        # The actual prompt for generation is the last user message
        if not gemini_history or gemini_history[-1]['role'] != 'user':
             print("Error: Last message for Gemini generate_content was not 'user'. History:", gemini_history)
             # Attempt to use the last message regardless if it exists
             if gemini_history:
                  last_user_prompt = gemini_history[-1]['parts'][0]
                  gemini_history = gemini_history[:-1] # History is everything *before* the last message
             else:
                  return "AI Error: No valid user prompt found in conversation history."
        else:
             last_user_prompt = gemini_history[-1]['parts'][0]
             gemini_history = gemini_history[:-1] # History is everything *before* the last user message


        # Start chat session or generate content
        # Add system instruction if available
        model_instance = genai.GenerativeModel(
             'gemini-1.5-flash-latest', # Or 1.5-pro if needed, flash is faster/cheaper
             generation_config=generation_config,
             system_instruction=(system_prompt if system_prompt else "You are a helpful academic advisor.")
             )

        if gemini_history:
             # Ensure history roles are correct ('user', 'model')
             valid_history = [m for m in gemini_history if m.get('role') in ['user', 'model']]
             chat = model_instance.start_chat(history=valid_history)
             response = chat.send_message(last_user_prompt)
        else:
             # If no history, just send the last user prompt with system instruction
             response = model_instance.generate_content(last_user_prompt)


        if response.parts:
            return response.text
        else:
            # Check for block reason
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'
            print(f"Gemini generation blocked or empty. Reason: {block_reason}. Feedback: {response.prompt_feedback}")
            return f"AI Error: Generation failed or was blocked (Reason: {block_reason}). Please rephrase your query."
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        # Log the history that caused the error for debugging
        print("Gemini History at time of error:", gemini_history)
        return f"AI Error: API call failed ({e})"


# --- HELPER FOR AI INTERACTION ---
def get_ai_response(provider: str, api_key: str, messages: List[Dict[str, str]]) -> str:
    """Gets response from the selected AI provider."""
    # Define system prompt here
    system_prompt = """You are CMU Engineering Advisor Bot, an expert AI assistant helping prospective graduate students find suitable programs at Carnegie Mellon's College of Engineering.

Your Goals:
1.  Understand the student's academic background, research interests, career goals, preferred learning style (theoretical vs. applied), and desired degree (M.S. or Ph.D.).
2.  Ask clarifying questions if needed, but aim to gather enough information in 1-3 interactions.
3.  Once you have sufficient information, explicitly state that you will now search for matching programs based on the conversation. Example: "Okay, based on our conversation, I'll look for [Degree Level] programs related to [Keywords]..."
4.  After stating you will search, your *next* response MUST be the analysis of program descriptions provided to you separately. Do not engage in further conversation before providing the analysis.
5.  When analyzing provided program descriptions: Recommend the top 1-3 programs that BEST match the student's profile *solely* based on the text snippets provided.
6.  For each recommendation: Include Program Name, a concise summary derived *only* from the provided description, Contact Information (if present in the snippet), and the Source URL. Explain *why* it's a good fit, referencing specific points from the student's profile (from the conversation history) and the program description snippet.
7.  If no provided programs are a good match, clearly state that and explain why based on the descriptions.
8.  Be concise, helpful, and professional. Use markdown for formatting recommendations (e.g., bold program names, use bullet points for analysis).
"""
    # Ensure system prompt is the first message if not already present
    if not messages or messages[0].get("role") != "system":
        conversation = [{"role": "system", "content": system_prompt}] + messages
    else:
        # If system prompt is already there, make sure it's the latest one
        conversation = [{"role": "system", "content": system_prompt}] + [m for m in messages if m.get("role") != "system"]


    if provider == "DeepSeek":
        # DeepSeek often works better with the system prompt integrated into the first user message or implicitly.
        # Let's try sending it as OpenAI format expects.
        return _call_deepseek_chat_api(api_key, conversation)
    elif provider == "Google Gemini":
        # Gemini handles system prompt separately.
        return _call_gemini_chat_api(api_key, conversation)
    else:
        return "Error: Invalid AI provider selected."

# --- KEYWORD FILTERING (Simplified) ---
def filter_programs_by_keywords(keywords: List[str], df_programs: pd.DataFrame, degree_level: str) -> pd.DataFrame:
    """Filters DataFrame based on keywords and degree."""
    if not keywords or df_programs.empty:
        print("Keyword list empty or DataFrame empty, returning no filter results.")
        return pd.DataFrame()

    df_filtered = df_programs[df_programs['degree_type'] == degree_level].copy()
    if df_filtered.empty:
        print(f"No programs found for degree level: {degree_level}")
        return pd.DataFrame()

    # Create a lowercased search text column for efficiency
    df_filtered['search_text'] = (df_filtered['name'].astype(str) + ' ' + df_filtered['description'].astype(str)).str.lower()

    # Create a regex pattern for keywords - match whole words if possible, be generous
    # Example: 'robotics' becomes r'\brobotics\b'. Handle multi-word keywords.
    patterns = []
    for k in keywords:
         # Escape special regex characters in the keyword
         escaped_k = re.escape(k)
         # Try to match whole word, fallback to substring if it contains spaces etc.
         if ' ' not in k and '-' not in k:
              patterns.append(r'\b' + escaped_k + r'\b')
         else: # For multi-word or hyphenated, just do substring match
              patterns.append(escaped_k)

    # Combine patterns with OR (|)
    keyword_regex = '|'.join(patterns)

    # Apply the regex search
    try:
        matches_mask = df_filtered['search_text'].str.contains(keyword_regex, case=False, regex=True, na=false)
        print(f"Filtering resulted in {matches_mask.sum()} matches.")
        return df_filtered[matches_mask]
    except Exception as e:
        print(f"Error during regex filtering: {e}. Falling back to simple 'in' check.")
        # Fallback to simple 'in' check if regex fails
        def check_match_simple(text):
            return any(k in text for k in keywords)
        matches_mask_simple = df_filtered['search_text'].apply(check_match_simple)
        print(f"Fallback filtering resulted in {matches_mask_simple.sum()} matches.")
        return df_filtered[matches_mask_simple]


# --- MAIN APPLICATION ---
def main():
    st.title("🤖 CMU Engineering AI Advisor")

    # --- Sidebar for Provider Selection ---
    with st.sidebar:
        st.image("https://www.cmu.edu/brand/brand-guidelines/assets/images/wordmarks-and-initials/cmu-wordmark-stacked-r-c.png", width='stretch')
        st.header("Configuration")
        available_providers = []
        # Check secrets more robustly
        if "DEEPSEEK_API_KEY" in st.secrets and st.secrets.DEEPSEEK_API_KEY:
             available_providers.append("DeepSeek")
        if "GEMINI_API_KEY" in st.secrets and st.secrets.GEMINI_API_KEY:
             available_providers.append("Google Gemini")

        if not available_providers:
            st.error("No valid AI provider API key found in Streamlit Secrets."); st.stop()

        default_provider = "Google Gemini" if "Google Gemini" in available_providers else available_providers[0]
        # Ensure default index is valid
        default_idx = available_providers.index(default_provider) if default_provider in available_providers else 0
        ai_provider = st.selectbox("Choose AI Model", available_providers, index=default_idx)
        st.caption(f"Using {ai_provider} for conversation and analysis.")
        # Add button to clear cache for debugging
        if st.button("Clear App Cache"):
             st.cache_data.clear()
             st.cache_resource.clear()
             st.success("App cache cleared. Please refresh the page.")


    # Get API key based on selection
    api_key_name = "DEEPSEEK_API_KEY" if ai_provider == "DeepSeek" else "GEMINI_API_KEY"
    api_key = st.secrets.get(api_key_name)
    if not api_key:
        st.error(f"{ai_provider} API key not found or empty in Streamlit Secrets."); st.stop()

    # --- Load Data ---
    @st.cache_resource # Cache the loaded dataframe for the session
    def load_data():
        with st.spinner("Loading CMU program information... (may take a minute on first run)"):
             df = get_cmu_program_data()
        # No need to stop here, empty check happens in get_cmu_program_data
        return df
    df_programs = load_data()
    # Check if df_programs is None or empty after loading attempt
    if df_programs is None or df_programs.empty:
         st.error("Failed to load program data after scraping. Cannot continue.")
         st.stop()


    # --- Initialize Chat ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm the CMU Engineering AI Advisor. To help you find the best graduate program, could you tell me a bit about:\n\n1.  Your **academic background** (e.g., undergrad major)?\n2.  Specific **research interests or keywords** (e.g., robotics, AI, sustainability)?\n3.  Your **career goals** (e.g., industry, academia, startup)?\n4.  Are you looking for an **M.S.** or a **Ph.D.**?"}]
    if "stage" not in st.session_state:
         st.session_state.stage = "gathering_info" # Stages: gathering_info, searching, presenting_results

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat Input ---
    if prompt := st.chat_input("Your response..."):
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            # --- AI LOGIC ---
            # 1. Send current conversation history to AI
            ai_raw_response = get_ai_response(ai_provider, api_key, st.session_state.messages)

            # 2. Check if AI indicates it's ready to search (more robust check)
            search_trigger_phrases = ["i will search", "let me look for", "finding programs",
                                      "based on that, i can search", "now i will find programs",
                                      "okay, searching now", "based on our conversation, i'll look"]
            # Also check if the AI's response seems short and might be just an acknowledgement before searching
            is_likely_search_trigger = any(phrase in ai_raw_response.lower() for phrase in search_trigger_phrases) or \
                                       (len(ai_raw_response.split()) < 30 and st.session_state.stage == "gathering_info")

            if is_likely_search_trigger and st.session_state.stage == "gathering_info":
                st.session_state.stage = "searching"
                message_placeholder.markdown(f"{ai_raw_response}\n\nOkay, extracting key info to find relevant programs...") # Show AI ack

                # --- Extract Keywords and Degree using AI ---
                keyword_extraction_prompt = f"""Review the following conversation history between an Advisor Bot and a student. Extract two pieces of information:
1. The desired **degree level** (Must be exactly "M.S." or "Ph.D.").
2. A comma-separated list of 5-10 specific **keywords** representing the student's core academic/research interests mentioned. Prioritize technical terms or specific fields.

Conversation History:
{"\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])}

---
Output Format (Exactly two lines):
Degree Level
Keyword1, Keyword2, Keyword3, ...
"""
                extraction_payload = [{"role":"user", "content": keyword_extraction_prompt}]
                # Use low temp for extraction
                extracted_info_raw = get_ai_response(ai_provider, api_key, extraction_payload)
                print("\n--- Keyword Extraction Attempt ---")
                print("Raw extraction response:", extracted_info_raw)

                # --- Parse Extraction Results ---
                degree_level_extracted = "M.S." # Default
                keywords_list = []
                try:
                    if "AI Error" not in extracted_info_raw:
                        lines = extracted_info_raw.strip().split('\n')
                        potential_degree = lines[0].strip().upper().replace('.', '') # Standardize (MS or PHD)
                        if potential_degree in ["MS", "M S"]:
                             degree_level_extracted = "M.S."
                        elif potential_degree == "PHD":
                             degree_level_extracted = "Ph.D."
                        else:
                             print(f"Warning: AI extracted unclear degree '{lines[0].strip()}', defaulting to M.S.")

                        if len(lines) > 1:
                            keywords_extracted_str = lines[1].strip()
                            keywords_list = [k.strip().lower() for k in keywords_extracted_str.split(',') if k.strip() and len(k.strip()) > 2] # Require keyword len > 2
                            if not keywords_list: print("Warning: AI extracted an empty keyword list.")
                        else: print("Warning: AI extraction did not provide keywords line.")
                    else: print("Keyword extraction failed due to AI error.")

                except Exception as e:
                     print(f"Error parsing AI keyword extraction: {e}. Raw: {extracted_info_raw}")
                     # Fallback: Try simple scan of conversation for degree
                     for msg in reversed(st.session_state.messages):
                          content_lower = msg["content"].lower()
                          if "ph.d." in content_lower or "phd" in content_lower or "doctoral" in content_lower: degree_level_extracted = "Ph.D."; break
                          if "m.s." in content_lower or "ms" in content_lower or "master" in content_lower: degree_level_extracted = "M.S."; break
                     print(f"Fallback degree detection: {degree_level_extracted}")


                # --- Filter Programs ---
                if not keywords_list:
                     message_placeholder.markdown(f"Okay, I'll search for **{degree_level_extracted}** programs broadly based on our conversation...")
                     # If no keywords, just filter by degree and maybe pass the whole conversation summary later?
                     # For now, just filter by degree
                     filtered_df = df_programs[df_programs['degree_type'] == degree_level_extracted].copy()
                else:
                     message_placeholder.markdown(f"Okay, searching for **{degree_level_extracted}** programs related to: *{', '.join(keywords_list)}*...")
                     filtered_df = filter_programs_by_keywords(keywords_list, df_programs, degree_level_extracted)


                # --- Prepare Context for Final Analysis ---
                if filtered_df.empty:
                    final_response = f"I couldn't find any CMU Engineering **{degree_level_extracted}** programs matching the keywords '{', '.join(keywords_list)}' based on the scraped data. Perhaps we could try different keywords or broaden the search?"
                    st.session_state.stage = "gathering_info" # Go back to gathering info
                    print("Filtering resulted in zero programs.")
                else:
                    message_placeholder.markdown(f"Found {len(filtered_df)} potential programs. Analyzing the best fits...")
                    print(f"Found {len(filtered_df)} potential programs after filtering.")
                    # Prepare context from filtered programs
                    context_for_ai = ""
                    # Use a token limit appropriate for the chosen model, Flash has 1M but keep it reasonable for cost/speed
                    token_limit = 8000
                    current_tokens = 0

                    # Include program snippets in the context
                    program_snippets = []
                    for _, row in filtered_df.iterrows(): # Consider more than just head if result count is low
                         program_info = f"### Program: {row['name']}\n**Degree:** {row['degree_type']}\n**URL:** {row['url']}\n**Contact:** {row['contact']}\n\n**Description Snippet:**\n{row['description']}\n\n---\n\n"
                         program_tokens = len(program_info.split()) # Rough estimate
                         if current_tokens + program_tokens < token_limit:
                              program_snippets.append(program_info)
                              current_tokens += program_tokens
                         else:
                              print(f"Context token limit ({token_limit}) reached. Stopped adding programs.")
                              break # Stop adding context

                    context_for_ai = "".join(program_snippets)

                    if not context_for_ai:
                         final_response = "Although programs were found, I couldn't assemble enough context to send for analysis within token limits. This might be a data issue."
                         st.session_state.stage = "gathering_info"
                         print("Context for AI analysis was empty after filtering.")
                    else:
                        # --- Final AI Call for Recommendations ---
                        recommendation_prompt = f"""Based on the complete conversation history provided below and the following relevant program description snippets extracted from the CMU Engineering website, please act as the CMU Engineering Advisor Bot.

Your task is to recommend the top 1-3 programs for the student and explain why, following the detailed instructions in your initial system prompt. Remember to base your analysis *strictly* on the provided snippets.

**Full Conversation History:**
{"\n".join([f"**{m['role'].upper()}**: {m['content']}" for m in st.session_state.messages])}

---
**Filtered Program Description Snippets (Analyze ONLY these):**
{context_for_ai}
---

**Your Recommendation and Analysis:**
"""
                        # Add the user prompt that triggered the search and the AI's acknowledgement to the history for the final call
                        final_payload = st.session_state.messages + [{"role": "assistant", "content": ai_raw_response}, {"role": "user", "content": recommendation_prompt}]

                        print("\n--- Final Recommendation Call ---")
                        # print("Payload Messages:", final_payload) # Debugging payload
                        final_response = get_ai_response(ai_provider, api_key, final_payload)
                        st.session_state.stage = "presenting_results" # Update stage
                        print("Recommendation generation complete.")

            else:
                # AI is asking follow-up or giving intermediate response
                final_response = ai_raw_response
                st.session_state.stage = "gathering_info" # Stay in info gathering

            # Display final AI response and add to history
            message_placeholder.markdown(final_response)
            # Avoid adding the massive recommendation prompt to history
            if st.session_state.stage != "presenting_results":
                 st.session_state.messages.append({"role": "assistant", "content": final_response})
            else:
                 # Add only the final recommendation text to history
                 st.session_state.messages.append({"role": "assistant", "content": final_response})


if __name__ == "__main__":
    main()
