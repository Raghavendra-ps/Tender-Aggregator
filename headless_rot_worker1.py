#!/usr/bin/env python3
# File: headless_rot_worker.py (Adapted from your stand.py for interactive CAPTCHA)

import asyncio
import base64
import argparse
import datetime # Ensure this is imported
from pathlib import Path
import httpx
import re
import logging
import json
from urllib.parse import urljoin, urlparse
from typing import Tuple, Optional, Dict, Set, List, Any

from playwright.async_api import async_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup, Tag

# --- URL_SUFFIX_ROT (Define it here if dashboard is to import it) ---
URL_SUFFIX_ROT = "?page=WebTenderStatusLists&service=page"

# --- Worker Configuration (Selectors - these are fine) ---
TENDER_STATUS_DROPDOWN_SELECTOR = "#tenderStatus"
CAPTCHA_IMAGE_SELECTOR = "#captchaImage"
CAPTCHA_TEXT_INPUT_SELECTOR = "#captchaText"
SEARCH_BUTTON_SELECTOR = "#Search"
RESULTS_TABLE_SELECTOR = "#tabList"
VIEW_STATUS_LINK_IN_RESULTS = "table#tabList tr[id^='informal'] a[id^='view']"
SUMMARY_LINK_ON_DETAIL_PAGE = "a#DirectLink_0:has-text('Click link to view the all stage summary Details')"
ERROR_MESSAGE_SELECTORS = ".error_message, .errormsg, #msgDiv"
PAGINATION_LINK_SELECTOR = "a#loadNext"

# --- Default Settings (Worker's internal fallbacks if not provided by dashboard) ---
MAX_ROT_LIST_PAGES_TO_FETCH_DEFAULT = 3
DETAIL_PROCESSING_CONCURRENCY_DEFAULT = 2
WAIT_AFTER_PAGINATION_CLICK_MS_DEFAULT = 7000
PAGE_LOAD_TIMEOUT_DEFAULT = 60000
DETAIL_PAGE_TIMEOUT_DEFAULT = 75000 # For View Status Link Navigation
POPUP_PAGE_TIMEOUT_DEFAULT = 60000  # For the final summary popup page
ELEMENT_TIMEOUT_DEFAULT = 20000
POST_SUBMIT_TIMEOUT_DEFAULT = 30000 # After CAPTCHA submission
CAPTCHA_ANSWER_WAIT_TIMEOUT_S = 300

# --- NEW DIRECTORY STRUCTURE CONSTANTS ---
try:
    PROJECT_ROOT = Path(__file__).parent.resolve()
except NameError:
    PROJECT_ROOT = Path('.').resolve()

SITE_DATA_ROOT = PROJECT_ROOT / "site_data"

# ROT Specific Output Paths (Worker writes here)
ROT_DATA_DIR = SITE_DATA_ROOT / "ROT"
ROT_MERGED_SITE_SPECIFIC_DIR = ROT_DATA_DIR / "MergedSiteSpecific"
ROT_DETAIL_HTMLS_DIR = ROT_DATA_DIR / "DetailHtmls" # NEW: For ROT_{TENDER_ID}_{DATETIME}_StageSummary.html

# TEMP Specific Paths (Worker uses these for communication and temporary files)
TEMP_DATA_DIR = SITE_DATA_ROOT / "TEMP"
TEMP_WORKER_RUNS_DIR = TEMP_DATA_DIR / "WorkerRuns" # Replaces old TEMP_WORKER_RUNS_DIR
TEMP_DEBUG_SCREENSHOTS_DIR = TEMP_DATA_DIR / "DebugScreenshots"
TEMP_CAPTCHA_IMAGES_DIR = TEMP_DATA_DIR / "CaptchaImages" # For any physical CAPTCHA images if saved

# LOGS Path
LOGS_BASE_DIR_WORKER = PROJECT_ROOT / "LOGS"
# --- END NEW DIRECTORY STRUCTURE CONSTANTS ---

SCRIPT_NAME_TAG = Path(__file__).stem
TODAY_STR = datetime.date.today().strftime("%Y-%m-%d")


# --- Logging Setup (Worker Specific) ---
logger = logging.getLogger(f"{SCRIPT_NAME_TAG}_worker") # Default logger name

def setup_worker_logging(site_key_for_log: str, run_id_for_log: str):
    global logger # Allow modification of the module-level logger
    logger_name = f"{SCRIPT_NAME_TAG}_worker_{site_key_for_log}_{run_id_for_log[-6:]}" # More unique name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers(): # Clear existing handlers for this specific logger instance if re-called
        logger.handlers.clear()

    # New log path
    log_dir_for_this_worker = LOGS_BASE_DIR_WORKER / "rot_worker" / site_key_for_log
    log_dir_for_this_worker.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir_for_this_worker / f"worker_{run_id_for_log}_{TODAY_STR}.log"

    # File Handler
    fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter(f"%(asctime)s [%(levelname)s] (ROTWorker-{site_key_for_log} Run-{run_id_for_log[-6:]}) %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    # Console Handler (optional for worker, good for direct debugging)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(f"[ROTWorker-{site_key_for_log} Run-{run_id_for_log[-6:]}] %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    logger.info(f"Worker logging initialized. Log file: {log_file_path}")

# --- Utility Functions (from your stand.py) ---
def _remove_html_protection(html_content: str) -> str:
    modified_content = html_content
    protection_script_pattern1 = re.compile(r'<script language="javascript" type="text/javascript">\s*if\s*\(\s*window\.name\s*==\s*""\s*\)\s*\{[^}]*window\.location\.replace[^}]*\}[^<]*else if\s*\(\s*window\.name\.indexOf\("Popup"\)\s*==\s*-1\)\s*\{[^}]*window\.location\.replace[^}]*\}\s*</script>', re.IGNORECASE | re.DOTALL)
    modified_content = protection_script_pattern1.sub("<!-- Protection Script 1 Removed -->", modified_content)
    if "Protection Script 1 Removed" not in modified_content:
        logger.debug("Specific script pattern 1 not found. Trying general patterns.")
        modified_content = re.sub(r'<script[^>]*>.*window\.name\s*==\s*""[^<]*?</script>', "<!-- Protection Script (window.name == \"\") Removed -->", modified_content, flags=re.IGNORECASE | re.DOTALL)
        modified_content = re.sub(r'<script[^>]*>.*window\.name\.indexOf\("Popup"\)\s*==\s*-1[^<]*?</script>', "<!-- Protection Script (indexOf Popup == -1) Removed -->", modified_content, flags=re.IGNORECASE | re.DOTALL)
    noscript_redirect_pattern = re.compile(r'<noscript>\s*<meta http-equiv="Refresh"[^>]*?UnauthorisedAccessPage[^>]*?>\s*</noscript>', re.IGNORECASE | re.DOTALL)
    modified_content = noscript_redirect_pattern.sub("<!-- Noscript Redirect Removed -->", modified_content)
    def remove_onpageshow(match):
        body_tag = match.group(0); cleaned_body_tag = re.sub(r'\s*onpageshow\s*=\s*".*?"', '', body_tag, flags=re.IGNORECASE); cleaned_body_tag = re.sub(r"\s*onpageshow\s*=\s*'.*?'", '', cleaned_body_tag, flags=re.IGNORECASE); return cleaned_body_tag
    modified_content = re.sub(r'<body[^>]*>', remove_onpageshow, modified_content, count=1, flags=re.IGNORECASE)
    if modified_content != html_content: logger.info("Applied HTML protection removal.")
    return modified_content

def clean_filename_rot(filename: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|\r\n\t]+', '_', filename); cleaned = re.sub(r'[\s_]+', '_', cleaned).strip('_'); return cleaned[:150] if len(cleaned) > 150 else cleaned

def update_worker_status(run_dir: Path, status_message: str):
    try:
        status_file = run_dir / "status.txt"
        status_file.write_text(status_message, encoding='utf-8')
        logger.info(f"Worker status updated to: {status_message}")
    except Exception as e:
        logger.error(f"Error writing worker status '{status_message}' to {run_dir}: {e}")

def _clean_path_component_worker(component: str) -> str:
    """
    Removes potentially unsafe characters from a path component, aiming to match
    dashboard.py's _clean_path_component for directory naming consistency.
    Allows alphanumeric, underscore, hyphen, period. Removes others (like spaces).
    """
    global logger # Access the module-level logger
    if not isinstance(component, str):
        if logger: logger.warning(f"Attempted to clean non-string path component: {type(component)}")
        else: print(f"Warning (_clean_path_component_worker): Attempted to clean non-string path component: {type(component)}")
        return ""
    # This regex removes anything NOT a word character (alphanumeric + underscore), hyphen, or period.
    # Crucially, it removes spaces.
    cleaned = re.sub(r'[^\w.\-]', '', component)
    if cleaned == "." or cleaned == "..": # Prevent relative path components
        if logger: logger.warning(f"Path component cleaned to unsafe value '{cleaned}', returning empty string.")
        else: print(f"Warning (_clean_path_component_worker): Path component cleaned to unsafe value '{cleaned}', returning empty string.")
        return ""
    return cleaned

async def _save_captcha_for_dashboard(page: Page, run_dir: Path, tender_status_to_select: str) -> bool:
    """Saves CAPTCHA image as base64 for the dashboard and signals readiness."""
    captcha_b64_file = run_dir / "captcha.b64"
    try:
        logger.info(f"Selecting tender status '{tender_status_to_select}'.")
        await page.locator(TENDER_STATUS_DROPDOWN_SELECTOR).select_option(value=tender_status_to_select, timeout=ELEMENT_TIMEOUT_DEFAULT)
        await page.wait_for_timeout(3000) # Allow dynamic CAPTCHA to load

        captcha_element = page.locator(CAPTCHA_IMAGE_SELECTOR)
        await captcha_element.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_DEFAULT)
        captcha_image_bytes = await captcha_element.screenshot(type='png')

        if not captcha_image_bytes:
            logger.error("CAPTCHA screenshot resulted in empty bytes.")
            update_worker_status(run_dir, "ERROR_CAPTCHA_SCREENSHOT_EMPTY")
            return False

        captcha_base64_data = base64.b64encode(captcha_image_bytes).decode('utf-8')
        captcha_b64_file.write_text(captcha_base64_data, encoding='utf-8')
        logger.info(f"CAPTCHA base64 data saved to: {captcha_b64_file}")
        update_worker_status(run_dir, "WAITING_CAPTCHA") # Signal dashboard CAPTCHA is ready
        return True
    except PlaywrightTimeout:
        logger.error("Timeout interacting with CAPTCHA elements.")
        update_worker_status(run_dir, "ERROR_CAPTCHA_ELEMENT_TIMEOUT")
        await page.screenshot(path=run_dir / "debug_captcha_timeout.png")
        return False
    except Exception as e:
        logger.error(f"Error saving CAPTCHA for dashboard: {e}", exc_info=True)
        update_worker_status(run_dir, f"ERROR_SAVING_CAPTCHA_{type(e).__name__}")
        await page.screenshot(path=run_dir / "debug_captcha_error.png")
        return False

async def _wait_for_captcha_solution(run_dir: Path) -> Optional[str]:
    answer_file = run_dir / "answer.txt"
    status_file = run_dir / "status.txt" # Dashboard updates this after writing answer
    logger.info(f"Waiting for CAPTCHA solution in {answer_file} (max {CAPTCHA_ANSWER_WAIT_TIMEOUT_S}s)...")
    
    for _ in range(CAPTCHA_ANSWER_WAIT_TIMEOUT_S): # Check every second
        if answer_file.is_file():
            try:
                # Optional: Check status file for a specific signal from dashboard
                # if status_file.is_file() and status_file.read_text(encoding='utf-8').strip() == "CAPTCHA_ANSWER_SUBMITTED_BY_DASHBOARD":
                solution = answer_file.read_text(encoding='utf-8').strip()
                if solution:
                    logger.info("CAPTCHA solution received.")
                    # answer_file.unlink(missing_ok=True) # Clean up answer file
                    return solution
                # else: # File exists but is empty, wait
            except Exception as e:
                logger.error(f"Error reading answer file {answer_file}: {e}")
                update_worker_status(run_dir, "ERROR_READING_ANSWER_FILE")
                return None # Propagate error
        await asyncio.sleep(1)

    logger.warning("Timeout waiting for CAPTCHA solution.")
    update_worker_status(run_dir, "TIMEOUT_CAPTCHA_INPUT")
    return None

def clean_text(text: Optional[str]) -> str: # Copied from scrape.py or filter_engine.py
    """Cleans text by replacing multiple whitespaces and stripping."""
    if text is None or text == "N/A": return "N/A"
    # Ensure input is treated as string, replace multiple whitespace with single space
    return re.sub(r'\s+', ' ', str(text)).strip()

def _format_rot_tags(data_dict: Dict[str, Any], site_key: str) -> str:
    """Formats a dictionary of ROT tender data into a tagged block string."""
    # Based on filter_engine.py TAG_REGEX for ROT
    # "RotSNo", "RotTenderId", "RotTitleRef", "RotOrganisationChain", "RotTenderStage",
    # "SourceSiteKey", "RotStatusDetailPageLink",
    # "RotStageSummaryFileStatus", "RotStageSummaryFilePath", "RotStageSummaryFilename"

    tags = ["--- TENDER START ---"]
    
    # Data from list page extraction
    tags.append(f"<RotSNo>{clean_text(data_dict.get('s_no_on_page', 'N/A'))}</RotSNo>")
    tags.append(f"<RotTenderId>{clean_text(data_dict.get('tender_id', 'N/A'))}</RotTenderId>")
    tags.append(f"<RotTitleRef>{clean_text(data_dict.get('title_and_ref_no', 'N/A'))}</RotTitleRef>")
    tags.append(f"<RotOrganisationChain>{clean_text(data_dict.get('organisation_chain', 'N/A'))}</RotOrganisationChain>")
    tags.append(f"<RotTenderStage>{clean_text(data_dict.get('tender_stage', 'N/A'))}</RotTenderStage>")
    tags.append(f"<SourceSiteKey>{clean_text(site_key)}</SourceSiteKey>")
    
    detail_page_link = data_dict.get('view_status_link_full_url', 'N/A') # Assuming this key is added
    tags.append(f"<RotStatusDetailPageLink>{clean_text(detail_page_link)}</RotStatusDetailPageLink>")

    # Data from detail processing (summary file)
    tags.append(f"<RotStageSummaryFileStatus>{clean_text(data_dict.get('summary_file_status', 'not_processed'))}</RotStageSummaryFileStatus>")
    tags.append(f"<RotStageSummaryFilePath>{clean_text(data_dict.get('summary_file_path_relative', 'N/A'))}</RotStageSummaryFilePath>")
    tags.append(f"<RotStageSummaryFilename>{clean_text(data_dict.get('summary_file_name', 'N/A'))}</RotStageSummaryFilename>")
    
    tags.append("--- TENDER END ---")
    return "\n".join(tag for tag in tags if tag is not None and tag.strip() and "N/A</" not in tag) # Filter out empty or "N/A" only tags

async def _handle_final_popup_content(
    popup_page: Page, site_key: str, tender_file_id: str, run_id: str
) -> Tuple[str, Optional[str], Optional[str]]:
    cleaned_site_key_for_dir = _clean_path_component_worker(site_key)
    site_specific_html_output_dir = ROT_DETAIL_HTMLS_DIR / cleaned_site_key_for_dir
    site_specific_html_output_dir.mkdir(parents=True, exist_ok=True)

    debug_screenshot_path_base = TEMP_DEBUG_SCREENSHOTS_DIR / "worker" / cleaned_site_key_for_dir / run_id / "popup_handling"
    debug_screenshot_path_base.mkdir(parents=True, exist_ok=True)

    log_prefix_popup = f"[{clean_filename_rot(tender_file_id)} Popup]"
    logger.info(f"{log_prefix_popup} Processing final popup page: {popup_page.url}")

    try:
        await popup_page.wait_for_load_state("domcontentloaded", timeout=POPUP_PAGE_TIMEOUT_DEFAULT)
        html_content_report_raw = await popup_page.content()

        if html_content_report_raw:
            logger.debug(f"{log_prefix_popup} Starting HTML cleanup.")
            soup = BeautifulSoup(html_content_report_raw, 'html.parser')

            # 1. Perform all cleanups on the full soup object first
            for tag_type in ['script', 'style', 'link']: # Remove script, style, and external link tags
                for tag in soup.find_all(tag_type):
                    tag.decompose()
            logger.debug(f"{log_prefix_popup} Removed script, style, and link tags.")

            # Remove common "Print" buttons/links more carefully
            print_elements_found = 0
            # Try by ID first (case-insensitive search for 'print' in id)
            for print_el_id in soup.find_all(id=re.compile(r'print', re.I)):
                print_el_id.decompose()
                print_elements_found += 1
            # Then by typical print href
            for print_el_href in soup.find_all('a', href=re.compile(r'javascript:window.print\(\)', re.I)):
                # Attempt to remove a simple parent container if it seems to only hold the print link
                parent = print_el_href.find_parent()
                if parent and parent.name in ['td', 'p', 'span', 'div'] and \
                   len(parent.get_text(strip=True)) == len(print_el_href.get_text(strip=True)) and \
                   len(parent.find_all(True, recursive=False)) == 1: # Parent has only this link as child element
                    parent.decompose()
                else:
                    print_el_href.decompose()
                print_elements_found += 1
            if print_elements_found > 0:
                logger.debug(f"{log_prefix_popup} Removed {print_elements_found} potential print-related element(s).")

            # Remove body attributes like onload, onpageshow if body exists
            if soup.body:
                if 'onload' in soup.body.attrs:
                    del soup.body['onload']
                    logger.debug(f"{log_prefix_popup} Removed onload attribute from body.")
                if 'onpageshow' in soup.body.attrs:
                    del soup.body['onpageshow']
                    logger.debug(f"{log_prefix_popup} Removed onpageshow attribute from body.")
            
            # 2. Decide which part of the cleaned soup to save
            html_to_save_str: str
            main_content_div = soup.find('div', id='printDisplayArea')
            if not main_content_div: main_content_div = soup.find('div', class_='Table')
            if not main_content_div: main_content_div = soup.find('div', class_='table')
            if not main_content_div: main_content_div = soup.find('div', class_='border')

            if main_content_div:
                html_to_save_str = str(main_content_div)
                logger.debug(f"{log_prefix_popup} Extracted content from a main container div for saving.")
            elif soup.body:
                # If we save the body, we take its inner HTML to avoid nested <body> tags
                # when dashboard's parser re-parses this fragment.
                html_to_save_str = "".join(str(c) for c in soup.body.contents)
                logger.debug(f"{log_prefix_popup} Saving inner HTML of body content.")
            else:
                html_to_save_str = str(soup) # Fallback to full cleaned soup
                logger.debug(f"{log_prefix_popup} Saving full cleaned soup as no specific body/main div found.")
            
            # 3. Apply script protection removal (which operates on strings)
            processed_html_content = _remove_html_protection(html_to_save_str)
            
            safe_id_part = clean_filename_rot(tender_file_id)
            timestamp_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename_html = f"ROT_{safe_id_part}_{timestamp_suffix}_StageSummary.html"
            save_path_html_absolute = site_specific_html_output_dir / save_filename_html
            
            save_path_html_absolute.write_text(processed_html_content, encoding='utf-8', errors='replace')
            
            try:
                save_path_html_relative = save_path_html_absolute.relative_to(ROT_DETAIL_HTMLS_DIR)
            except ValueError:
                logger.warning(f"{log_prefix_popup} Could not make path relative for {save_path_html_absolute} against {ROT_DETAIL_HTMLS_DIR}. Storing absolute path: {save_path_html_absolute}")
                save_path_html_relative = save_path_html_absolute
            
            logger.info(f"{log_prefix_popup} ✅ CLEANED & SAVED HTML: {save_filename_html} to {site_specific_html_output_dir}")
            return "downloaded", str(save_path_html_relative), save_filename_html
        else:
            logger.warning(f"{log_prefix_popup} No HTML content received from popup page.")
            return "download_no_content", None, None
    except PlaywrightTimeout: # ... (rest of except blocks are fine) ...
        logger.error(f"{log_prefix_popup} Timeout waiting for popup content.", exc_info=False)
        screenshot_file = debug_screenshot_path_base / f"DEBUG_POPUP_TIMEOUT_{clean_filename_rot(tender_file_id)}_{SCRIPT_NAME_TAG}.png"
        if popup_page and not popup_page.is_closed(): await popup_page.screenshot(path=screenshot_file)
        logger.info(f"Debug screenshot saved to {screenshot_file}")
        return "download_timeout", None, None
    except Exception as e:
        logger.error(f"{log_prefix_popup} Error during HTML cleanup or saving for popup: {e}", exc_info=True)
        screenshot_file = debug_screenshot_path_base / f"DEBUG_POPUP_ERROR_{clean_filename_rot(tender_file_id)}_{SCRIPT_NAME_TAG}.png"
        if popup_page and not popup_page.is_closed(): await popup_page.screenshot(path=screenshot_file)
        logger.info(f"Debug screenshot saved to {screenshot_file}")
        return f"download_error_{type(e).__name__}", None, None

def parse_rot_summary_html(html_file_path: Path) -> Dict[str, Any]:
    extracted_data = {
        "original_filename": html_file_path.name, # Store for reference
        "page_title": "ROT Summary", # Default title
        "key_details": {}, # For specific key-value pairs
        "sections": []     # For larger blocks of content (e.g., tables)
                           # Each item can be a dict: {"title": "Section Title", "html_content": "..."}
    }
    if not html_file_path.is_file():
        logger.error(f"ROT HTML summary file not found for parsing: {html_file_path}")
        extracted_data["error_message"] = "Summary HTML file not found on server."
        return extracted_data

    try:
        with open(html_file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # If the saved content was just a fragment (e.g., a div's content),
        # BeautifulSoup can parse it directly. If it was saved as a full HTML doc string,
        # it will also work.
        soup = BeautifulSoup(content, 'html.parser')

        # Attempt to find a main title (if the saved HTML fragment doesn't have <title>)
        # Look for prominent headers within the content.
        h1 = soup.find(['h1', 'h2', 'h3', 'div'], class_=re.compile(r'(title|header|heading)', re.I))
        if h1:
            extracted_data["page_title"] = h1.get_text(strip=True)
        elif soup.title and soup.title.string: # If full HTML was saved
             extracted_data["page_title"] = soup.title.string.strip()


        # --- CUSTOM PARSING LOGIC STARTS HERE ---
        # This needs to be tailored to the structure of YOUR cleaned ROT summary HTMLs.
        # Inspect your saved HTML files from site_data/ROT/DetailHtmls/{SITE_KEY}/...

        # Example 1: Extracting key-value pairs from a definition list or simple table structure
        # Suppose your HTML has: <td><strong>Tender ID:</strong></td><td>12345</td>
        
        # Find all potential label cells (strong, th, or td with specific class)
        label_tags = soup.find_all(['strong', 'th', 'td'], class_=lambda x: x and 'label' in x.lower())
        if not label_tags: # Fallback to any strong or th
            label_tags = soup.find_all(['strong', 'th'])

        for label_tag in label_tags:
            key_text = label_tag.get_text(strip=True).replace(':', '').strip()
            value_tag = label_tag.find_next_sibling(['td', 'dd']) # Common next tags
            if not value_tag: # If label and value are in same tag, e.g. <strong>Label:</strong> Value
                parent_text = label_tag.parent.get_text(separator='|', strip=True)
                if '|' in parent_text: # Heuristic
                    parts = parent_text.split('|', 1)
                    if key_text.lower() in parts[0].lower() and len(parts) > 1:
                        value_text = parts[1].strip()
                        if key_text and value_text: extracted_data["key_details"][key_text] = value_text
                        continue # Move to next label
            
            if value_tag:
                value_text = value_tag.get_text(strip=True)
                if key_text and value_text: # Only add if both key and value are found
                    # Avoid overly generic keys or very long values for key_details
                    if len(key_text) < 50 and len(value_text) < 200:
                         extracted_data["key_details"][key_text] = value_text

        # Example 2: Extracting all tables as separate sections
        tables = soup.find_all('table')
        if tables:
            for i, table in enumerate(tables):
                # Try to find a preceding header for the table to use as a title
                table_title = f"Data Table {i+1}"
                prev_sibling = table.find_previous_sibling(['h1', 'h2', 'h3', 'h4', 'p'])
                if prev_sibling and len(prev_sibling.get_text(strip=True)) < 100: # Heuristic for a title
                    table_title = prev_sibling.get_text(strip=True)
                
                # Basic table cleanup (remove nested scripts/styles if any survived worker cleanup)
                for s in table.find_all(['script', 'style']): s.decompose()
                
                extracted_data["sections"].append({
                    "title": table_title,
                    "html_content": str(table) # Convert table back to HTML string
                })
        elif not extracted_data["key_details"]: # If no tables and no key details, add the whole soup
             extracted_data["sections"].append({
                    "title": "Full Summary Content",
                    "html_content": str(soup) # The entire cleaned HTML fragment
                })


        if not extracted_data["key_details"] and not extracted_data["sections"]:
             extracted_data["error_message"] = "Could not extract structured details or sections from the summary HTML."

    except Exception as e:
        logger.error(f"Error parsing ROT summary HTML content from {html_file_path}: {e}", exc_info=True)
        extracted_data["error_message"] = f"An error occurred while parsing the summary content: {e}"
    
    return extracted_data

async def process_single_tender_detail_page(
    semaphore: asyncio.Semaphore, main_browser_context: BrowserContext,
    detail_page_url: str, site_key: str,
    tender_display_id: str, list_page_num: int,
    run_id: str
) -> Tuple[str, Optional[str], Optional[str]]:

    log_prefix = f"[ListPage{list_page_num}_{clean_filename_rot(tender_display_id)}]"
    logger.info(f"{log_prefix} Starting detail processing for URL: {detail_page_url}")

    detail_page: Optional[Page] = None
    popup_page: Optional[Page] = None

    # Use the new cleaner for the site_key part of the debug directory path
    cleaned_site_key_for_dir = _clean_path_component_worker(site_key)

    debug_screenshot_path_base_detail = TEMP_DEBUG_SCREENSHOTS_DIR / "worker" / cleaned_site_key_for_dir / run_id / "detail_page_processing" / clean_filename_rot(tender_display_id)
    debug_screenshot_path_base_detail.mkdir(parents=True, exist_ok=True)

    async with semaphore:
        try:
            detail_page = await main_browser_context.new_page()
            await detail_page.goto(detail_page_url, wait_until="domcontentloaded", timeout=DETAIL_PAGE_TIMEOUT_DEFAULT)
            logger.info(f"{log_prefix} Navigated to detail page: {detail_page.url}")

            summary_link_locator = detail_page.locator(SUMMARY_LINK_ON_DETAIL_PAGE).first
            await summary_link_locator.wait_for(state="visible", timeout=ELEMENT_TIMEOUT_DEFAULT)
            logger.info(f"{log_prefix} Final summary link found. Clicking...")

            async with detail_page.expect_popup(timeout=POPUP_PAGE_TIMEOUT_DEFAULT) as popup_info:
                await summary_link_locator.click()
            popup_page = await popup_info.value

            # Pass run_id to _handle_final_popup_content
            status, rel_path, fname = await _handle_final_popup_content(
                popup_page, site_key, tender_display_id,
                run_id
            )

            if status == "downloaded":
                logger.info(f"{log_prefix} Successfully processed detail and popup. File: {fname}")
            else:
                logger.warning(f"{log_prefix} Failed to process content from popup. Status: {status}")
            return status, rel_path, fname

        except PlaywrightTimeout as pt_err:
            logger.error(f"{log_prefix} Timeout in detail/popup processing for '{detail_page_url}': {pt_err}", exc_info=False)
            if detail_page and not detail_page.is_closed():
                screenshot_file_detail = debug_screenshot_path_base_detail / f"DETAIL_PG_TIMEOUT.png"
                await detail_page.screenshot(path=screenshot_file_detail)
                logger.info(f"Debug screenshot (detail page timeout) saved to {screenshot_file_detail}")
            if popup_page and not popup_page.is_closed():
                screenshot_file_popup = debug_screenshot_path_base_detail / f"POPUP_PG_TIMEOUT_OUTER.png"
                await popup_page.screenshot(path=screenshot_file_popup)
                logger.info(f"Debug screenshot (popup page timeout outer) saved to {screenshot_file_popup}")
            return f"detail_processing_timeout", None, None
        except Exception as e:
            logger.error(f"{log_prefix} Error processing detail link '{detail_page_url}': {e}", exc_info=True)
            if detail_page and not detail_page.is_closed():
                screenshot_file_detail_err = debug_screenshot_path_base_detail / f"DETAIL_PG_ERROR.png"
                await detail_page.screenshot(path=screenshot_file_detail_err)
                logger.info(f"Debug screenshot (detail page error) saved to {screenshot_file_detail_err}")
            return f"detail_processing_error_{type(e).__name__}", None, None
        finally:
            if popup_page and not popup_page.is_closed(): await popup_page.close()
            if detail_page and not detail_page.is_closed(): await detail_page.close()

async def extract_table_data_from_current_list_page(
    page: Page, list_page_num: int, site_base_url_for_links: str,
    site_key: str, run_id: str # <<<< NEW: Added site_key and run_id
) -> List[Dict[str, Any]]:
    extracted_data: list[dict[str, Any]] = []
    logger.info(f"[ListPage{list_page_num}] Extracting table data...")

    # NEW: Structured debug screenshot path for table extraction issues
    debug_table_extract_dir = TEMP_DEBUG_SCREENSHOTS_DIR / "worker" / site_key / run_id / "table_extraction_errors"
    debug_table_extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        results_table_html = await page.locator(RESULTS_TABLE_SELECTOR).inner_html(timeout=ELEMENT_TIMEOUT_DEFAULT)
        soup = BeautifulSoup(results_table_html, "html.parser"); rows = soup.select("tr[id^='informal']")
        if not rows:
            tbl_txt_el = soup.select_one(RESULTS_TABLE_SELECTOR);
            if tbl_txt_el and ("no records found" in tbl_txt_el.get_text(strip=True).lower() or "no tenders available" in tbl_txt_el.get_text(strip=True).lower()):
                logger.info(f"[ListPage{list_page_num}] 'No records' text found in table."); return [{"status_signal": "NO_RECORDS_FOUND_IN_TABLE"}]
            logger.warning(f"[ListPage{list_page_num}] No data rows (tr[id^='informal']) found."); return []
        for row_idx, row_tag in enumerate(rows):
            cols = row_tag.find_all("td")
            if len(cols) == 6:
                s_no = cols[0].get_text(strip=True); tender_id = cols[1].get_text(strip=True)
                title_ref = cols[2].get_text(strip=True); org_chain = cols[3].get_text(strip=True)
                tender_stage = cols[4].get_text(strip=True); view_link_href_relative = "N/A"
                link_tag = cols[5].find("a", id=re.compile(r"^view_?\d*$"))
                if link_tag and link_tag.has_attr('href'):
                    href_val = link_tag['href'].strip()
                    if href_val and href_val != '#': view_link_href_relative = href_val
                
                full_detail_url = "N/A"
                if view_link_href_relative != "N/A":
                     full_detail_url = urljoin(site_base_url_for_links, view_link_href_relative)

                extracted_data.append({
                    "list_page_num": str(list_page_num), "s_no_on_page": s_no, "tender_id": tender_id,
                    "title_and_ref_no": title_ref, "organisation_chain": org_chain, "tender_stage": tender_stage,
                    "view_status_link_href_relative": view_link_href_relative,
                    "view_status_link_full_url": full_detail_url
                })
            else: logger.warning(f"[ListPage{list_page_num}] Row {row_idx+1} has {len(cols)} cols, expected 6. Skipping.")
        logger.info(f"[ListPage{list_page_num}] Extracted {len(extracted_data)} items.")
    except PlaywrightTimeout:
        logger.error(f"[ListPage{list_page_num}] Timeout getting HTML of '{RESULTS_TABLE_SELECTOR}'.")
        screenshot_file = debug_table_extract_dir / f"TABLE_EXTRACT_TIMEOUT_P{list_page_num}_{SCRIPT_NAME_TAG}.png"
        await page.screenshot(path=screenshot_file)
        logger.info(f"Debug screenshot (table extract timeout) saved to {screenshot_file}")
    except Exception as e:
        logger.error(f"[ListPage{list_page_num}] Error extracting table data: {e}", exc_info=True)
        screenshot_file = debug_table_extract_dir / f"TABLE_EXTRACT_ERROR_P{list_page_num}_{SCRIPT_NAME_TAG}.png"
        await page.screenshot(path=screenshot_file)
        logger.info(f"Debug screenshot (table extract error) saved to {screenshot_file}")
    return extracted_data

async def run_headless_rot_scrape_orchestration(
    site_key: str, run_id: str,
    target_site_search_url: str, target_site_base_url: str, tender_status_to_select: str,
    worker_settings: Dict[str, Any]
):
    run_specific_temp_dir = TEMP_WORKER_RUNS_DIR / run_id
    run_specific_temp_dir.mkdir(parents=True, exist_ok=True)

    setup_worker_logging(site_key, run_id) # This uses new LOGS_BASE_DIR_WORKER
    logger.info(f"--- Starting Headless ROT Worker for Site: {site_key}, Run ID: {run_id} ---")
    logger.info(f"Target Search URL: {target_site_search_url}")
    logger.info(f"Target Base URL for links: {target_site_base_url}")
    logger.info(f"Tender Status to Select: {tender_status_to_select}")
    logger.info(f"Worker Settings: {worker_settings}")

    ROT_MERGED_SITE_SPECIFIC_DIR.mkdir(parents=True, exist_ok=True)
    final_site_tagged_output_path = ROT_MERGED_SITE_SPECIFIC_DIR / f"Merged_ROT_{clean_filename_rot(site_key)}_{TODAY_STR}.txt"

    update_worker_status(run_specific_temp_dir, "WORKER_STARTED")

    cleaned_site_key_for_paths = _clean_path_component_worker(site_key) # Use the consistent cleaner
    orchestration_debug_screenshots_dir = TEMP_DEBUG_SCREENSHOTS_DIR / "worker" / cleaned_site_key_for_paths / run_id / "orchestration_errors"
    orchestration_debug_screenshots_dir.mkdir(parents=True, exist_ok=True)

    was_successful_overall_scrape = False # Flag to determine if consolidation trigger should happen
    page: Optional[Page] = None

    async with async_playwright() as playwright:
        browser = None
        context = None
        try:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context(ignore_https_errors=True, accept_downloads=True)
            page = await context.new_page()

            update_worker_status(run_specific_temp_dir, "FETCHING_CAPTCHA")
            logger.info(f"Navigating to: {target_site_search_url}")
            await page.goto(target_site_search_url, wait_until="domcontentloaded", timeout=worker_settings.get("page_load_timeout", PAGE_LOAD_TIMEOUT_DEFAULT))

            if not await _save_captcha_for_dashboard(page, run_specific_temp_dir, tender_status_to_select):
                logger.error(f"Failed to save CAPTCHA for dashboard (Site: {site_key}, Run: {run_id}). Aborting worker.")
                # Status already set by _save_captcha_for_dashboard on error
                return # Exit

            captcha_solution = await _wait_for_captcha_solution(run_specific_temp_dir)
            if not captcha_solution:
                logger.error(f"No CAPTCHA solution received or timeout (Site: {site_key}, Run: {run_id}). Aborting worker.")
                # Status already set by _wait_for_captcha_solution on error/timeout
                return # Exit

            update_worker_status(run_specific_temp_dir, "PROCESSING_WITH_CAPTCHA")
            logger.info(f"CAPTCHA solution received (Site: {site_key}, Run: {run_id}). Submitting form...")
            await page.locator(CAPTCHA_TEXT_INPUT_SELECTOR).fill(captcha_solution)
            await page.locator(SEARCH_BUTTON_SELECTOR).click()

            logger.info(f"Waiting for initial results ('{RESULTS_TABLE_SELECTOR}') OR error (Site: {site_key}, Run: {run_id}).")
            try:
                await page.wait_for_selector(f"{RESULTS_TABLE_SELECTOR}, {ERROR_MESSAGE_SELECTORS}", state="visible", timeout=worker_settings.get("post_submit_timeout", POST_SUBMIT_TIMEOUT_DEFAULT))
            except PlaywrightTimeout:
                logger.error(f"Timeout: Neither results table nor error message appeared after CAPTCHA submission (Site: {site_key}, Run: {run_id}).")
                await page.screenshot(path=orchestration_debug_screenshots_dir / f"debug_captcha_submit_overall_timeout.png")
                update_worker_status(run_specific_temp_dir, "ERROR_CAPTCHA_SUBMIT_TIMEOUT")
                return # Exit

            if not await page.locator(RESULTS_TABLE_SELECTOR).is_visible(timeout=2000):
                err_text = "Unknown (Results table not found)"
                err_loc = page.locator(ERROR_MESSAGE_SELECTORS).first
                if await err_loc.count() > 0 and await err_loc.is_visible(timeout=1000):
                    err_text_content = await err_loc.text_content()
                    err_text = err_text_content.strip() if err_text_content else err_text
                logger.error(f"CAPTCHA submission failed or error on page: {err_text} (Site: {site_key}, Run: {run_id})")
                await page.screenshot(path=orchestration_debug_screenshots_dir / f"debug_captcha_submit_error.png")
                update_worker_status(run_specific_temp_dir, f"ERROR_CAPTCHA_SUBMIT_{clean_filename_rot(err_text[:50])}")
                return # Exit

            logger.info(f"✅ Initial results table detected after CAPTCHA (Site: {site_key}, Run: {run_id}). Proceeding with scrape.")
            update_worker_status(run_specific_temp_dir, "SCRAPING_RESULTS")

            processed_list_pages = 0
            detail_semaphore = asyncio.Semaphore(worker_settings.get("detail_concurrency", DETAIL_PROCESSING_CONCURRENCY_DEFAULT))
            consecutive_empty_list_pages = 0
            max_list_pages = worker_settings.get("max_list_pages", MAX_ROT_LIST_PAGES_TO_FETCH_DEFAULT)
            total_tenders_processed_successfully = 0

            with open(final_site_tagged_output_path, "a", encoding="utf-8") as outfile_merged:
                while processed_list_pages < max_list_pages:
                    processed_list_pages += 1
                    logger.info(f"--- Processing List Page: {processed_list_pages} (Site: {site_key}, Run: {run_id}) ---")
                    update_worker_status(run_specific_temp_dir, f"PROCESSING_LIST_PAGE_{processed_list_pages}")
                    await page.wait_for_selector(RESULTS_TABLE_SELECTOR, state="visible", timeout=worker_settings.get("element_timeout", ELEMENT_TIMEOUT_DEFAULT))

                    current_page_items_raw = await extract_table_data_from_current_list_page(
                        page, processed_list_pages, target_site_base_url,
                        site_key, run_id
                    )

                    if current_page_items_raw and isinstance(current_page_items_raw[0], dict) and current_page_items_raw[0].get("status_signal") == "NO_RECORDS_FOUND_IN_TABLE":
                        logger.info(f"'No records' signal on page {processed_list_pages} (Site: {site_key}, Run: {run_id}). Ending scrape.")
                        update_worker_status(run_specific_temp_dir, "FINISHED_NO_DATA_ON_PAGE")
                        was_successful_overall_scrape = True
                        break

                    if not current_page_items_raw:
                        logger.info(f"No items found on list page {processed_list_pages} (Site: {site_key}, Run: {run_id}).")
                        consecutive_empty_list_pages += 1
                        if consecutive_empty_list_pages >= 2:
                            logger.warning(f"Two consecutive empty list pages (Site: {site_key}, Run: {run_id}). Ending scrape.")
                            update_worker_status(run_specific_temp_dir, "FINISHED_EMPTY_PAGES")
                            was_successful_overall_scrape = True
                            break
                    else:
                        consecutive_empty_list_pages = 0
                        detail_tasks = []
                        for item_data_list_page in current_page_items_raw:
                            if not isinstance(item_data_list_page, dict):
                                logger.warning(f"Skipping non-dict item from list page: {item_data_list_page}")
                                continue
                            detail_url_from_list = item_data_list_page.get("view_status_link_full_url")
                            tender_id_from_list = item_data_list_page.get("tender_id", f"Item_P{processed_list_pages}_R{len(detail_tasks)+1}")

                            if detail_url_from_list and detail_url_from_list != "N/A":
                                task = process_single_tender_detail_page(
                                    detail_semaphore, context, detail_url_from_list,
                                    site_key, tender_id_from_list, processed_list_pages,
                                    run_id
                                )
                                detail_tasks.append((task, item_data_list_page))
                            else:
                                augmented_item = item_data_list_page.copy()
                                augmented_item['summary_file_status'] = "no_detail_link"
                                tagged_block = _format_rot_tags(augmented_item, site_key)
                                if tagged_block: outfile_merged.write(tagged_block + "\n\n")

                        if detail_tasks:
                            logger.info(f"Gathering {len(detail_tasks)} detail tasks for list page {processed_list_pages} (Site: {site_key}, Run: {run_id})...")
                            gathered_detail_results = await asyncio.gather(*(task_tuple[0] for task_tuple in detail_tasks))
                            logger.info(f"All detail tasks for list page {processed_list_pages} finished (Site: {site_key}, Run: {run_id}).")
                            for i, detail_result_tuple in enumerate(gathered_detail_results):
                                original_list_item_data = detail_tasks[i][1]
                                augmented_item_data = original_list_item_data.copy()
                                status, rel_path, fname = detail_result_tuple
                                augmented_item_data['summary_file_status'] = status
                                augmented_item_data['summary_file_path_relative'] = rel_path if rel_path else "N/A"
                                augmented_item_data['summary_file_name'] = fname if fname else "N/A"
                                if status == "downloaded":
                                    total_tenders_processed_successfully +=1
                                tagged_block = _format_rot_tags(augmented_item_data, site_key)
                                if tagged_block:
                                    outfile_merged.write(tagged_block + "\n\n")

                    next_page_locator = page.locator(PAGINATION_LINK_SELECTOR)
                    if await next_page_locator.is_visible(timeout=5000):
                        if processed_list_pages >= max_list_pages:
                            logger.info(f"Reached max list pages ({max_list_pages}) (Site: {site_key}, Run: {run_id}). Stopping.")
                            update_worker_status(run_specific_temp_dir, "FINISHED_MAX_PAGES_REACHED")
                            was_successful_overall_scrape = True
                            break
                        logger.info(f"Clicking 'Next' for list page {processed_list_pages + 1} (Site: {site_key}, Run: {run_id}).")
                        await next_page_locator.click()
                        await page.wait_for_timeout(worker_settings.get('pagination_wait', WAIT_AFTER_PAGINATION_CLICK_MS_DEFAULT))
                    else:
                        logger.info(f"No 'Next' page link found after list page {processed_list_pages} (Site: {site_key}, Run: {run_id}). End of results.")
                        update_worker_status(run_specific_temp_dir, "FINISHED_END_OF_RESULTS")
                        was_successful_overall_scrape = True
                        break
            
            current_status_on_file = ""
            if (run_specific_temp_dir / "status.txt").is_file():
                current_status_on_file = (run_specific_temp_dir / "status.txt").read_text().strip()

            if not current_status_on_file.startswith(("FINISHED_", "ERROR_")):
                update_worker_status(run_specific_temp_dir, "FINISHED_SUCCESS")
            
            # If any of the break conditions that set a "FINISHED_" status occurred,
            # or if the loop completed normally (now FINISHED_SUCCESS), mark as successful overall.
            if (run_specific_temp_dir / "status.txt").read_text().strip().startswith("FINISHED_"):
                 was_successful_overall_scrape = True


            logger.info(f"ROT scrape processing finished for site {site_key}, Run ID {run_id}. Successfully processed HTML summaries for {total_tenders_processed_successfully} tenders.")
            logger.info(f"Final site-specific merged ROT data for {site_key} is in: {final_site_tagged_output_path}")

        except Exception as e:
            logger.critical(f"Major error in ROT worker for site {site_key}, Run ID {run_id}: {e}", exc_info=True)
            update_worker_status(run_specific_temp_dir, f"ERROR_UNHANDLED_{type(e).__name__}")
            if page and not page.is_closed():
                await page.screenshot(path=orchestration_debug_screenshots_dir / f"debug_worker_critical_error.png")
            was_successful_overall_scrape = False # Ensure this is false on major error
        finally:
            if context: await context.close()
            if browser: await browser.close()
            logger.info(f"Playwright resources for ROT worker (Site: {site_key}, Run: {run_id}) closed.")
            
            final_status_check = (run_specific_temp_dir / "status.txt").read_text(encoding='utf-8').strip() if (run_specific_temp_dir / "status.txt").is_file() else "UNKNOWN_FINAL"
            logger.info(f"Worker for {site_key} (Run: {run_id}) terminated with status: {final_status_check}")

            # MODIFIED: Trigger dashboard consolidation endpoint only if the scrape was generally successful
            if was_successful_overall_scrape: # Check the flag
                dashboard_consolidation_url = "http://localhost:8081/trigger-rot-consolidation-from-worker" # Configurable if needed
                logger.info(f"Attempting to trigger ROT consolidation at dashboard: {dashboard_consolidation_url} (Site: {site_key}, Run: {run_id})")
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        response = await client.post(dashboard_consolidation_url)
                        if response.status_code == 200:
                            logger.info(f"Successfully triggered ROT consolidation via dashboard: {response.text} (Site: {site_key}, Run: {run_id})")
                        else:
                            logger.error(f"Failed to trigger ROT consolidation. Dashboard responded with {response.status_code}: {response.text} (Site: {site_key}, Run: {run_id})")
                except httpx.RequestError as exc_http:
                     logger.error(f"HTTP request error trying to trigger ROT consolidation: {exc_http} (Site: {site_key}, Run: {run_id})")
                except Exception as e_trigger:
                    logger.error(f"Unexpected error trying to trigger ROT consolidation: {e_trigger} (Site: {site_key}, Run: {run_id})", exc_info=True)
            else:
                logger.info(f"Skipping ROT consolidation trigger for site {site_key}, Run ID {run_id}, as the run was not marked successful (Final Status: {final_status_check}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless ROT Worker with interactive CAPTCHA.")
    parser.add_argument("--site_key", required=True, help="Site key (e.g., CPPP, Assam) from settings.json.")
    parser.add_argument("--run_id", required=True, help="Unique ID for this worker run, provided by dashboard.")
    parser.add_argument("--site_url", required=True, help="Full URL of the ROT search page for the site.")
    parser.add_argument("--base_url", required=True, help="Base URL for resolving relative links for the site.")
    parser.add_argument("--tender_status", required=True, help="Tender status value to select.")
    # Optional: Allow passing worker settings as JSON string
    parser.add_argument("--settings_json", type=str, help="JSON string of worker-specific settings (max_pages, timeouts etc.)")

    args = parser.parse_args()

    worker_run_settings = { # Defaults
        "max_list_pages": MAX_ROT_LIST_PAGES_TO_FETCH_DEFAULT,
        "detail_concurrency": DETAIL_PROCESSING_CONCURRENCY_DEFAULT,
        "pagination_wait": WAIT_AFTER_PAGINATION_CLICK_MS_DEFAULT,
        "page_load_timeout": PAGE_LOAD_TIMEOUT_DEFAULT,
        "detail_page_timeout": DETAIL_PAGE_TIMEOUT_DEFAULT,
        "popup_page_timeout": POPUP_PAGE_TIMEOUT_DEFAULT,
        "element_timeout": ELEMENT_TIMEOUT_DEFAULT,
        "post_submit_timeout": POST_SUBMIT_TIMEOUT_DEFAULT
    }
    if args.settings_json:
        try:
            passed_settings = json.loads(args.settings_json)
            worker_run_settings.update(passed_settings)
        except json.JSONDecodeError:
            print(f"WARNING: Could not parse --settings_json: {args.settings_json}. Using defaults.")


    print(f"--- Headless ROT Worker Initializing ---")
    print(f"Site Key: {args.site_key}, Run ID: {args.run_id}, Tender Status: {args.tender_status}")
    print(f"Using Search URL: {args.site_url}, Base URL: {args.base_url}")
    print(f"Effective Worker Settings: {worker_run_settings}")
    print(f"Communication Dir: {TEMP_WORKER_RUNS_DIR / args.run_id}")
    print(f"Final HTML Output Base: {ROT_DETAIL_HTMLS_DIR / args.site_key}")
    print(f"Final Tagged Data Output File (Append Mode): {ROT_MERGED_SITE_SPECIFIC_DIR / f'Merged_ROT_{clean_filename_rot(args.site_key)}_{TODAY_STR}.txt'}")
    print("-" * 70)

    # Create base directories if they don't exist
    TEMP_WORKER_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ROT_DETAIL_HTMLS_DIR.mkdir(parents=True, exist_ok=True)
    ROT_MERGED_SITE_SPECIFIC_DIR.mkdir(parents=True, exist_ok=True)
    
    asyncio.run(run_headless_rot_scrape_orchestration(
        args.site_key,
        args.run_id,
        args.site_url,
        args.base_url,
        args.tender_status,
        worker_run_settings
    ))
    print(f"--- Headless ROT Worker for {args.site_key} (Run ID: {args.run_id}) Finished ---")
