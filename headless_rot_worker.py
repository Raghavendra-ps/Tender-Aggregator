#!/usr/bin/env python3
# File: headless_rot_worker.py (Adapted from your stand.py for interactive CAPTCHA)

import os 
import asyncio
import base64
import argparse
import datetime 
from pathlib import Path
import httpx 
import re
import logging
import json
from urllib.parse import urljoin, urlparse
from typing import Tuple, Optional, Dict, Set, List, Any
import sys # For sys.executable

from playwright.async_api import async_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup, Tag

# --- URL_SUFFIX_ROT ---
URL_SUFFIX_ROT = "?page=WebTenderStatusLists&service=page"

# --- Worker Configuration (Selectors) ---
TENDER_STATUS_DROPDOWN_SELECTOR = "#tenderStatus"
CAPTCHA_IMAGE_SELECTOR = "#captchaImage"
CAPTCHA_TEXT_INPUT_SELECTOR = "#captchaText"
SEARCH_BUTTON_SELECTOR = "#Search"
RESULTS_TABLE_SELECTOR = "#tabList"
VIEW_STATUS_LINK_IN_RESULTS = "table#tabList tr[id^='informal'] a[id^='view']" # Not directly used, but good to keep
SUMMARY_LINK_ON_DETAIL_PAGE = "a#DirectLink_0:has-text('Click link to view the all stage summary Details')"
ERROR_MESSAGE_SELECTORS = ".error_message, .errormsg, #msgDiv"
PAGINATION_LINK_SELECTOR = "a#loadNext"

# --- Default Settings (Worker's internal fallbacks if not provided by dashboard) ---
MAX_ROT_LIST_PAGES_TO_FETCH_DEFAULT = 3
DETAIL_PROCESSING_CONCURRENCY_DEFAULT = 2
WAIT_AFTER_PAGINATION_CLICK_MS_DEFAULT = 7000
PAGE_LOAD_TIMEOUT_DEFAULT = 60000
DETAIL_PAGE_TIMEOUT_DEFAULT = 75000 
POPUP_PAGE_TIMEOUT_DEFAULT = 60000  
ELEMENT_TIMEOUT_DEFAULT = 20000
POST_SUBMIT_TIMEOUT_DEFAULT = 30000 
CAPTCHA_ANSWER_WAIT_TIMEOUT_S = 300

# --- Directory Structure Constants ---
try:
    PROJECT_ROOT = Path(__file__).parent.resolve()
except NameError: # Fallback for environments where __file__ might not be defined
    PROJECT_ROOT = Path('.').resolve()

SITE_DATA_ROOT = PROJECT_ROOT / "site_data"
ROT_DATA_DIR = SITE_DATA_ROOT / "ROT"
ROT_MERGED_SITE_SPECIFIC_DIR = ROT_DATA_DIR / "MergedSiteSpecific"
ROT_DETAIL_HTMLS_DIR = ROT_DATA_DIR / "DetailHtmls"
# NEW: Directory for structured JSON output from ROT HTMLs
ROT_STRUCTURED_JSON_DIR = SITE_DATA_ROOT / "AI" / "StructuredROTData" # As per your request

TEMP_DATA_DIR = SITE_DATA_ROOT / "TEMP"
TEMP_WORKER_RUNS_DIR = TEMP_DATA_DIR / "WorkerRuns" 
TEMP_DEBUG_SCREENSHOTS_DIR = TEMP_DATA_DIR / "DebugScreenshots"
# TEMP_CAPTCHA_IMAGES_DIR no longer used to save files, only for b64

LOGS_BASE_DIR_WORKER = PROJECT_ROOT / "LOGS"

SCRIPT_NAME_TAG = Path(__file__).stem
TODAY_STR = datetime.date.today().strftime("%Y-%m-%d")


# --- Logging Setup ---
logger = logging.getLogger(f"{SCRIPT_NAME_TAG}_placeholder") # Placeholder

def setup_worker_logging(site_key_for_log: str, run_id_for_log: str):
    global logger 
    logger_name = f"{SCRIPT_NAME_TAG}_worker_{site_key_for_log}_{run_id_for_log[-6:]}"
    logger = logging.getLogger(logger_name)
    
    if logger.hasHandlers(): 
        logger.handlers.clear()
        logger.propagate = False 

    logger.setLevel(logging.INFO)
    logger.propagate = False 
    
    log_dir_for_this_worker = LOGS_BASE_DIR_WORKER / "rot_worker" / site_key_for_log
    log_dir_for_this_worker.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir_for_this_worker / f"worker_{run_id_for_log}_{TODAY_STR}.log"

    fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter(f"%(asctime)s [%(levelname)s] (ROTWorker-{site_key_for_log} Run-{run_id_for_log[-6:]}) %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout) # Use sys.stdout for console
    ch.setFormatter(logging.Formatter(f"[ROTWorker-{site_key_for_log} Run-{run_id_for_log[-6:]}] %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    logger.info(f"Worker logging initialized. Log file: {log_file_path}")

# --- Utility Functions ---
def _remove_html_protection(html_content: str) -> str:
    modified_content = html_content
    protection_script_pattern1 = re.compile(r'<script language="javascript" type="text/javascript">\s*if\s*\(\s*window\.name\s*==\s*""\s*\)\s*\{[^}]*window\.location\.replace[^}]*\}[^<]*else if\s*\(\s*window\.name\.indexOf\("Popup"\)\s*==\s*-1\)\s*\{[^}]*window\.location\.replace[^}]*\}\s*</script>', re.IGNORECASE | re.DOTALL)
    modified_content = protection_script_pattern1.sub("<!-- Protection Script 1 Removed -->", modified_content)
    if "Protection Script 1 Removed" not in modified_content:
        if hasattr(logger, 'debug'): logger.debug("Specific script pattern 1 not found. Trying general patterns.")
        modified_content = re.sub(r'<script[^>]*>.*window\.name\s*==\s*""[^<]*?</script>', "<!-- Protection Script (window.name == \"\") Removed -->", modified_content, flags=re.IGNORECASE | re.DOTALL)
        modified_content = re.sub(r'<script[^>]*>.*window\.name\.indexOf\("Popup"\)\s*==\s*-1[^<]*?</script>', "<!-- Protection Script (indexOf Popup == -1) Removed -->", modified_content, flags=re.IGNORECASE | re.DOTALL)
    noscript_redirect_pattern = re.compile(r'<noscript>\s*<meta http-equiv="Refresh"[^>]*?UnauthorisedAccessPage[^>]*?>\s*</noscript>', re.IGNORECASE | re.DOTALL)
    modified_content = noscript_redirect_pattern.sub("<!-- Noscript Redirect Removed -->", modified_content)
    def remove_onpageshow(match):
        body_tag = match.group(0); cleaned_body_tag = re.sub(r'\s*onpageshow\s*=\s*".*?"', '', body_tag, flags=re.IGNORECASE); cleaned_body_tag = re.sub(r"\s*onpageshow\s*=\s*'.*?'", '', cleaned_body_tag, flags=re.IGNORECASE); return cleaned_body_tag
    modified_content = re.sub(r'<body[^>]*>', remove_onpageshow, modified_content, count=1, flags=re.IGNORECASE)
    if modified_content != html_content and hasattr(logger, 'info'): logger.info("Applied HTML protection removal.")
    return modified_content

def clean_filename_rot(filename: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|\r\n\t]+', '_', filename)
    cleaned = re.sub(r'[\s_]+', '_', cleaned).strip('_')
    return cleaned[:150] if len(cleaned) > 150 else cleaned

def update_worker_status(run_dir: Path, status_message: str):
    try:
        status_file = run_dir / "status.txt"
        status_file.write_text(status_message, encoding='utf-8')
        if hasattr(logger, 'info'): logger.info(f"Worker status updated to: {status_message}")
    except Exception as e:
        if hasattr(logger, 'error'): logger.error(f"Error writing worker status '{status_message}' to {run_dir}: {e}")

def _clean_path_component_worker(component: str) -> str:
    global logger 
    if not isinstance(component, str):
        if hasattr(logger, 'warning'): logger.warning(f"Attempted to clean non-string path component: {type(component)}")
        else: print(f"Warning (_clean_path_component_worker): Attempted to clean non-string path component: {type(component)}")
        return ""
    cleaned = re.sub(r'[^\w.\-]', '', component)
    if cleaned == "." or cleaned == "..": 
        if hasattr(logger, 'warning'): logger.warning(f"Path component cleaned to unsafe value '{cleaned}', returning empty string.")
        else: print(f"Warning (_clean_path_component_worker): Path component cleaned to unsafe value '{cleaned}', returning empty string.")
        return ""
    return cleaned

async def _save_captcha_for_dashboard(page: Page, run_dir: Path, tender_status_to_select: str, element_timeout: int) -> bool:
    captcha_b64_file = run_dir / "captcha.b64"
    debug_screenshots_captcha_dir = run_dir / "captcha_debug"
    debug_screenshots_captcha_dir.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"Selecting tender status '{tender_status_to_select}'.")
        await page.locator(TENDER_STATUS_DROPDOWN_SELECTOR).select_option(value=tender_status_to_select, timeout=element_timeout)
        await page.wait_for_timeout(3000) 
        captcha_element = page.locator(CAPTCHA_IMAGE_SELECTOR)
        await captcha_element.wait_for(state="visible", timeout=element_timeout)
        captcha_image_bytes = await captcha_element.screenshot(type='png')
        if not captcha_image_bytes:
            logger.error("CAPTCHA screenshot resulted in empty bytes.")
            update_worker_status(run_dir, "ERROR_CAPTCHA_SCREENSHOT_EMPTY")
            return False
        captcha_base64_data = base64.b64encode(captcha_image_bytes).decode('utf-8')
        captcha_b64_file.write_text(captcha_base64_data, encoding='utf-8')
        logger.info(f"CAPTCHA base64 data saved to: {captcha_b64_file}")
        update_worker_status(run_dir, "WAITING_CAPTCHA")
        return True
    except PlaywrightTimeout:
        logger.error(f"Timeout ({element_timeout}ms) interacting with CAPTCHA elements.")
        update_worker_status(run_dir, "ERROR_CAPTCHA_ELEMENT_TIMEOUT")
        if page and not page.is_closed(): await page.screenshot(path=debug_screenshots_captcha_dir / "debug_captcha_timeout.png")
        return False
    except Exception as e:
        logger.error(f"Error saving CAPTCHA for dashboard: {e}", exc_info=True)
        update_worker_status(run_dir, f"ERROR_SAVING_CAPTCHA_{type(e).__name__}")
        if page and not page.is_closed(): await page.screenshot(path=debug_screenshots_captcha_dir / "debug_captcha_error.png")
        return False

async def _wait_for_captcha_solution(run_dir: Path) -> Optional[str]:
    answer_file = run_dir / "answer.txt"
    logger.info(f"Waiting for CAPTCHA solution in {answer_file} (max {CAPTCHA_ANSWER_WAIT_TIMEOUT_S}s)...")
    for _ in range(CAPTCHA_ANSWER_WAIT_TIMEOUT_S): 
        if answer_file.is_file():
            try:
                solution = answer_file.read_text(encoding='utf-8').strip()
                if solution:
                    logger.info("CAPTCHA solution received.")
                    return solution
            except Exception as e:
                logger.error(f"Error reading answer file {answer_file}: {e}")
                update_worker_status(run_dir, "ERROR_READING_ANSWER_FILE")
                return None 
        await asyncio.sleep(1)
    logger.warning("Timeout waiting for CAPTCHA solution.")
    update_worker_status(run_dir, "TIMEOUT_CAPTCHA_INPUT")
    return None

def clean_text(text: Optional[str]) -> str:
    if text is None or text == "N/A": return "N/A"
    return re.sub(r'\s+', ' ', str(text)).strip()

def _format_rot_tags(data_dict: Dict[str, Any], site_key: str) -> str:
    tags = ["--- TENDER START ---"]
    tags.append(f"<RotSNo>{clean_text(data_dict.get('s_no_on_page', 'N/A'))}</RotSNo>")
    tags.append(f"<RotTenderId>{clean_text(data_dict.get('tender_id', 'N/A'))}</RotTenderId>")
    tags.append(f"<RotTitleRef>{clean_text(data_dict.get('title_and_ref_no', 'N/A'))}</RotTitleRef>")
    tags.append(f"<RotOrganisationChain>{clean_text(data_dict.get('organisation_chain', 'N/A'))}</RotOrganisationChain>")
    tags.append(f"<RotTenderStage>{clean_text(data_dict.get('tender_stage', 'N/A'))}</RotTenderStage>")
    tags.append(f"<SourceSiteKey>{clean_text(site_key)}</SourceSiteKey>")
    detail_page_link = data_dict.get('view_status_link_full_url', 'N/A')
    tags.append(f"<RotStatusDetailPageLink>{clean_text(detail_page_link)}</RotStatusDetailPageLink>")
    tags.append(f"<RotStageSummaryFileStatus>{clean_text(data_dict.get('summary_file_status', 'not_processed'))}</RotStageSummaryFileStatus>")
    tags.append(f"<RotStageSummaryFilePath>{clean_text(data_dict.get('summary_file_path_relative', 'N/A'))}</RotStageSummaryFilePath>")
    tags.append(f"<RotStageSummaryFilename>{clean_text(data_dict.get('summary_file_name', 'N/A'))}</RotStageSummaryFilename>")
    tags.append("--- TENDER END ---")
    # Ensure that "N/A" only tags are still included if other valid data exists,
    # but filter out blocks that would ONLY contain "N/A" tags (unlikely with this structure).
    # The previous join was fine.
    return "\n".join(tag for tag in tags if tag is not None and tag.strip())


async def _handle_final_popup_content(
    popup_page: Page, site_key: str, tender_file_id: str, run_id: str,
    popup_page_timeout: int 
) -> Tuple[str, Optional[str], Optional[str]]:
    cleaned_site_key_for_dir = _clean_path_component_worker(site_key)
    site_specific_html_output_dir = ROT_DETAIL_HTMLS_DIR / cleaned_site_key_for_dir
    site_specific_html_output_dir.mkdir(parents=True, exist_ok=True)

    # Changed base for popup screenshots to be per tender_file_id
    debug_screenshot_path_base = TEMP_DEBUG_SCREENSHOTS_DIR / "worker" / cleaned_site_key_for_dir / run_id / "popup_handling" / clean_filename_rot(tender_file_id)
    debug_screenshot_path_base.mkdir(parents=True, exist_ok=True)

    log_prefix_popup = f"[{clean_filename_rot(tender_file_id)} Popup]"
    logger.info(f"{log_prefix_popup} Processing final popup page: {popup_page.url}")

    try:
        await popup_page.wait_for_load_state("domcontentloaded", timeout=popup_page_timeout)
        html_content_report_raw = await popup_page.content()

        if html_content_report_raw:
            logger.debug(f"{log_prefix_popup} Starting HTML cleanup.")
            soup = BeautifulSoup(html_content_report_raw, 'html.parser')
            for tag_type in ['script', 'style', 'link']: 
                for tag in soup.find_all(tag_type): tag.decompose()
            
            print_elements_found = 0
            for print_el_id in soup.find_all(id=re.compile(r'print', re.I)): print_el_id.decompose(); print_elements_found +=1
            for print_el_href in soup.find_all('a', href=re.compile(r'javascript:window.print\(\)', re.I)):
                parent = print_el_href.find_parent()
                if parent and parent.name in ['td', 'p', 'span', 'div'] and \
                   len(parent.get_text(strip=True)) == len(print_el_href.get_text(strip=True)) and \
                   len(parent.find_all(True, recursive=False)) == 1: parent.decompose()
                else: print_el_href.decompose()
                print_elements_found += 1
            if print_elements_found > 0: logger.debug(f"{log_prefix_popup} Removed {print_elements_found} print elements.")
            
            if soup.body:
                if 'onload' in soup.body.attrs: del soup.body['onload']
                if 'onpageshow' in soup.body.attrs: del soup.body['onpageshow']
            
            main_content_container = soup.find('form', id='bidSummaryForm')
            if not main_content_container: main_content_container = soup.find('div', id='printDisplayArea')
            if not main_content_container: main_content_container = soup.find('div', class_='Table')
            if not main_content_container: main_content_container = soup.find('div', class_='table')
            if not main_content_container: main_content_container = soup.find('div', class_='border')

            html_to_save_str = str(main_content_container) if main_content_container else ("".join(str(c) for c in soup.body.contents) if soup.body else str(soup))
            
            if main_content_container: logger.debug(f"{log_prefix_popup} Extracted from specific container for saving.")
            elif soup.body: logger.debug(f"{log_prefix_popup} Saving inner HTML of body.")
            else: logger.debug(f"{log_prefix_popup} Saving full cleaned soup.")

            processed_html_content = _remove_html_protection(html_to_save_str)
            
            safe_id_part = clean_filename_rot(tender_file_id)
            timestamp_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename_html = f"ROT_{safe_id_part}_{timestamp_suffix}_StageSummary.html"
            save_path_html_absolute = site_specific_html_output_dir / save_filename_html
            
            save_path_html_absolute.write_text(processed_html_content, encoding='utf-8', errors='replace')
            
            try:
                save_path_html_relative = save_path_html_absolute.relative_to(ROT_DETAIL_HTMLS_DIR)
            except ValueError:
                logger.warning(f"{log_prefix_popup} Path not relative: {save_path_html_absolute} vs {ROT_DETAIL_HTMLS_DIR}. Using absolute.")
                save_path_html_relative = save_path_html_absolute
            
            logger.info(f"{log_prefix_popup} ✅ CLEANED & SAVED HTML: {save_filename_html} to {site_specific_html_output_dir}")

            # --- Call script to extract structured JSON ---
            structured_json_output_target_dir = ROT_STRUCTURED_JSON_DIR / cleaned_site_key_for_dir
            structured_json_output_target_dir.mkdir(parents=True, exist_ok=True)
            
            # Path to extract_structured_rot.py - THIS MUST BE CORRECT
            # Assuming it's in Tender-Aggregator/utils/
            path_to_extraction_script = PROJECT_ROOT / "utils" / "extract_structured_rot.py"
            
            python_executable_for_subprocess = sys.executable

            if path_to_extraction_script.is_file():
                cmd = [
                    python_executable_for_subprocess,
                    str(path_to_extraction_script),
                    str(save_path_html_absolute), # Input HTML file
                    "--output-dir", str(structured_json_output_target_dir),
                    "--site-key", site_key # Pass site_key to the script
                ]
                logger.info(f"{log_prefix_popup} Attempting to extract structured JSON: {' '.join(cmd)}")
                try:
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()

                    if process.returncode == 0:
                        logger.info(f"{log_prefix_popup} Successfully extracted structured JSON for {save_filename_html}.")
                        if stdout and stdout.strip(): logger.debug(f"{log_prefix_popup} Extractor STDOUT: {stdout.decode(errors='replace').strip()}")
                    else:
                        logger.error(f"{log_prefix_popup} Failed to extract structured JSON for {save_filename_html}. RC: {process.returncode}")
                        if stdout and stdout.strip(): logger.error(f"{log_prefix_popup} Extractor STDOUT: {stdout.decode(errors='replace').strip()}")
                        if stderr and stderr.strip(): logger.error(f"{log_prefix_popup} Extractor STDERR: {stderr.decode(errors='replace').strip()}")
                except Exception as e_struct_extract:
                    logger.error(f"{log_prefix_popup} Exception calling JSON extractor for {save_filename_html}: {e_struct_extract}", exc_info=True)
            else:
                logger.warning(f"{log_prefix_popup} Structured data extraction script NOT FOUND at {path_to_extraction_script}. Skipping JSON extraction.")
            # --- END NEW ---

            return "downloaded", str(save_path_html_relative), save_filename_html
        else:
            logger.warning(f"{log_prefix_popup} No HTML content received from popup page.")
            return "download_no_content", None, None
    except PlaywrightTimeout:
        logger.error(f"{log_prefix_popup} Timeout waiting for popup content ({popup_page_timeout}ms).")
        if popup_page and not popup_page.is_closed(): await popup_page.screenshot(path=debug_screenshot_path_base / f"POPUP_TIMEOUT.png")
        return "download_timeout", None, None
    except Exception as e:
        logger.error(f"{log_prefix_popup} Error in popup handling: {e}", exc_info=True)
        if popup_page and not popup_page.is_closed(): await popup_page.screenshot(path=debug_screenshot_path_base / f"POPUP_ERROR.png")
        return f"download_error_{type(e).__name__}", None, None


async def process_single_tender_detail_page(
    semaphore: asyncio.Semaphore, main_browser_context: BrowserContext,
    detail_page_url: str, site_key: str,
    tender_display_id: str, list_page_num: int,
    run_id: str,
    detail_page_timeout: int, element_timeout: int, popup_page_timeout_for_detail: int 
) -> Tuple[str, Optional[str], Optional[str]]:
    log_prefix = f"[ListPage{list_page_num}_{clean_filename_rot(tender_display_id)}]"
    logger.info(f"{log_prefix} Starting detail processing for URL: {detail_page_url}")
    detail_page: Optional[Page] = None; popup_page: Optional[Page] = None
    cleaned_site_key_for_dir = _clean_path_component_worker(site_key)
    debug_screenshot_path_base_detail = TEMP_DEBUG_SCREENSHOTS_DIR / "worker" / cleaned_site_key_for_dir / run_id / "detail_page_processing" / clean_filename_rot(tender_display_id)
    debug_screenshot_path_base_detail.mkdir(parents=True, exist_ok=True)

    async with semaphore:
        try:
            detail_page = await main_browser_context.new_page()
            await detail_page.goto(detail_page_url, wait_until="domcontentloaded", timeout=detail_page_timeout)
            logger.info(f"{log_prefix} Navigated to detail page: {detail_page.url}")

            summary_link_locator = detail_page.locator(SUMMARY_LINK_ON_DETAIL_PAGE).first
            await summary_link_locator.wait_for(state="visible", timeout=element_timeout)
            logger.info(f"{log_prefix} Final summary link found. Clicking...")

            async with detail_page.expect_popup(timeout=popup_page_timeout_for_detail) as popup_info:
                await summary_link_locator.click()
            popup_page = await popup_info.value
            
            status, rel_path, fname = await _handle_final_popup_content(
                popup_page, site_key, tender_display_id, run_id, 
                popup_page_timeout_for_detail # Pass the same timeout used for expect_popup
            )

            if status == "downloaded": logger.info(f"{log_prefix} Successfully processed detail and popup. File: {fname}")
            else: logger.warning(f"{log_prefix} Failed to process content from popup. Status: {status}")
            return status, rel_path, fname
        except PlaywrightTimeout as pt_err:
            logger.error(f"{log_prefix} Timeout in detail/popup ({detail_page_timeout}ms or {popup_page_timeout_for_detail}ms) processing for '{detail_page_url}': {pt_err}", exc_info=False)
            if detail_page and not detail_page.is_closed(): await detail_page.screenshot(path=debug_screenshot_path_base_detail / f"DETAIL_PG_TIMEOUT.png")
            if popup_page and not popup_page.is_closed(): await popup_page.screenshot(path=debug_screenshot_path_base_detail / f"POPUP_PG_TIMEOUT_OUTER.png")
            return f"detail_processing_timeout", None, None
        except Exception as e:
            logger.error(f"{log_prefix} Error processing detail link '{detail_page_url}': {e}", exc_info=True)
            if detail_page and not detail_page.is_closed(): await detail_page.screenshot(path=debug_screenshot_path_base_detail / f"DETAIL_PG_ERROR.png")
            return f"detail_processing_error_{type(e).__name__}", None, None
        finally:
            if popup_page and not popup_page.is_closed(): await popup_page.close()
            if detail_page and not detail_page.is_closed(): await detail_page.close()

async def extract_table_data_from_current_list_page(
    page: Page, list_page_num: int, site_base_url_for_links: str,
    site_key: str, run_id: str, element_timeout: int
) -> List[Dict[str, Any]]:
    extracted_data: List[Dict[str, Any]] = []
    logger.info(f"[ListPage{list_page_num}] Extracting table data...")
    # Use _clean_path_component_worker for site_key in path
    debug_table_extract_dir = TEMP_DEBUG_SCREENSHOTS_DIR / "worker" / _clean_path_component_worker(site_key) / run_id / "table_extraction_errors"
    debug_table_extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        results_table_html = await page.locator(RESULTS_TABLE_SELECTOR).inner_html(timeout=element_timeout)
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
                full_detail_url = urljoin(site_base_url_for_links, view_link_href_relative) if view_link_href_relative != "N/A" else "N/A"
                extracted_data.append({
                    "list_page_num": str(list_page_num), "s_no_on_page": s_no, "tender_id": tender_id,
                    "title_and_ref_no": title_ref, "organisation_chain": org_chain, "tender_stage": tender_stage,
                    "view_status_link_href_relative": view_link_href_relative,
                    "view_status_link_full_url": full_detail_url
                })
            else: logger.warning(f"[ListPage{list_page_num}] Row {row_idx+1} has {len(cols)} cols, expected 6.")
        logger.info(f"[ListPage{list_page_num}] Extracted {len(extracted_data)} items.")
    except PlaywrightTimeout:
        logger.error(f"[ListPage{list_page_num}] Timeout ({element_timeout}ms) getting HTML of '{RESULTS_TABLE_SELECTOR}'.")
        if page and not page.is_closed(): await page.screenshot(path=debug_table_extract_dir / f"TABLE_EXTRACT_TIMEOUT_P{list_page_num}.png")
    except Exception as e:
        logger.error(f"[ListPage{list_page_num}] Error extracting table data: {e}", exc_info=True)
        if page and not page.is_closed(): await page.screenshot(path=debug_table_extract_dir / f"TABLE_EXTRACT_ERROR_P{list_page_num}.png")
    return extracted_data

async def run_headless_rot_scrape_orchestration(
    site_key: str, run_id: str,
    target_site_search_url: str, target_site_base_url: str, tender_status_to_select: str,
    worker_settings: Dict[str, Any]
):
    run_specific_temp_dir = TEMP_WORKER_RUNS_DIR / run_id
    run_specific_temp_dir.mkdir(parents=True, exist_ok=True)
    setup_worker_logging(site_key, run_id) 
    logger.info(f"--- Starting Headless ROT Worker for Site: {site_key}, Run ID: {run_id} ---")
    logger.info(f"Settings from dashboard: {worker_settings}")

    # Ensure structured JSON base output directory exists for this worker run
    ROT_STRUCTURED_JSON_DIR.mkdir(parents=True, exist_ok=True)

    safe_cleaned_site_key_for_file = clean_filename_rot(site_key)
    final_site_tagged_output_path = ROT_MERGED_SITE_SPECIFIC_DIR / f"Merged_ROT_{safe_cleaned_site_key_for_file}_{TODAY_STR}.txt"
    update_worker_status(run_specific_temp_dir, "WORKER_STARTED")
    cleaned_site_key_for_paths = _clean_path_component_worker(site_key)
    orchestration_debug_screenshots_dir = TEMP_DEBUG_SCREENSHOTS_DIR / "worker" / cleaned_site_key_for_paths / run_id / "orchestration_errors"
    orchestration_debug_screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    was_successful_overall_scrape = False 
    page: Optional[Page] = None

    # Get all specific timeouts and limits from worker_settings, with global defaults
    page_load_timeout_ws = worker_settings.get("page_load_timeout", PAGE_LOAD_TIMEOUT_DEFAULT)
    detail_page_timeout_ws = worker_settings.get("detail_page_timeout", DETAIL_PAGE_TIMEOUT_DEFAULT)
    popup_page_timeout_ws = worker_settings.get("popup_page_timeout", POPUP_PAGE_TIMEOUT_DEFAULT) # For final summary popup
    element_timeout_ws = worker_settings.get("element_timeout", ELEMENT_TIMEOUT_DEFAULT)
    post_submit_timeout_ws = worker_settings.get("post_submit_timeout", POST_SUBMIT_TIMEOUT_DEFAULT)
    pagination_wait_ws = worker_settings.get("pagination_wait", WAIT_AFTER_PAGINATION_CLICK_MS_DEFAULT)
    detail_concurrency_ws = worker_settings.get("detail_concurrency", DETAIL_PROCESSING_CONCURRENCY_DEFAULT)
    max_list_pages_ws = worker_settings.get("max_list_pages", MAX_ROT_LIST_PAGES_TO_FETCH_DEFAULT)
    
    logger.info(f"Effective settings - PageLoadTO:{page_load_timeout_ws}, DetailPageTO:{detail_page_timeout_ws}, PopupPageTO:{popup_page_timeout_ws}, ElementTO:{element_timeout_ws}, PostSubmitTO:{post_submit_timeout_ws}, PagWait:{pagination_wait_ws}, DetailConc:{detail_concurrency_ws}, MaxListPgs:{max_list_pages_ws}")

    async with async_playwright() as playwright:
        browser = None; context = None
        try:
            browser = await playwright.chromium.launch(headless=True, args=['--no-sandbox'])
            context = await browser.new_context(ignore_https_errors=True, accept_downloads=True)
            page = await context.new_page()

            update_worker_status(run_specific_temp_dir, "FETCHING_CAPTCHA")
            await page.goto(target_site_search_url, wait_until="domcontentloaded", timeout=page_load_timeout_ws)

            if not await _save_captcha_for_dashboard(page, run_specific_temp_dir, tender_status_to_select, element_timeout_ws):
                return 
            captcha_solution = await _wait_for_captcha_solution(run_specific_temp_dir)
            if not captcha_solution: return 

            update_worker_status(run_specific_temp_dir, "PROCESSING_WITH_CAPTCHA")
            await page.locator(CAPTCHA_TEXT_INPUT_SELECTOR).fill(captcha_solution)
            await page.locator(SEARCH_BUTTON_SELECTOR).click()

            try:
                await page.wait_for_selector(f"{RESULTS_TABLE_SELECTOR}, {ERROR_MESSAGE_SELECTORS}", state="visible", timeout=post_submit_timeout_ws)
            except PlaywrightTimeout:
                logger.error(f"Timeout ({post_submit_timeout_ws}ms) post-CAPTCHA: Results/Error not found.")
                if page and not page.is_closed(): await page.screenshot(path=orchestration_debug_screenshots_dir / "debug_captcha_submit_timeout.png")
                update_worker_status(run_specific_temp_dir, "ERROR_CAPTCHA_SUBMIT_TIMEOUT")
                return 

            if not await page.locator(RESULTS_TABLE_SELECTOR).is_visible(timeout=2000):
                err_text = "Unknown (Results table not found post-submit)"
                err_loc = page.locator(ERROR_MESSAGE_SELECTORS).first
                if await err_loc.count() > 0 and await err_loc.is_visible(timeout=1000):
                    err_text_content = await err_loc.text_content(); err_text = err_text_content.strip() if err_text_content else err_text
                logger.error(f"CAPTCHA submit failed/error: {err_text}")
                if page and not page.is_closed(): await page.screenshot(path=orchestration_debug_screenshots_dir / f"debug_captcha_submit_err.png")
                update_worker_status(run_specific_temp_dir, f"ERROR_CAPTCHA_SUBMIT_{clean_filename_rot(err_text[:50])}")
                return 

            update_worker_status(run_specific_temp_dir, "SCRAPING_RESULTS")
            processed_list_pages = 0; detail_semaphore = asyncio.Semaphore(detail_concurrency_ws)
            consecutive_empty_list_pages = 0; total_tenders_processed_successfully = 0

            with open(final_site_tagged_output_path, "a", encoding="utf-8") as outfile_merged:
                while processed_list_pages < max_list_pages_ws:
                    processed_list_pages += 1
                    logger.info(f"--- Processing List Page: {processed_list_pages}/{max_list_pages_ws} (Site: {site_key}) ---")
                    update_worker_status(run_specific_temp_dir, f"PROCESSING_LIST_PAGE_{processed_list_pages}")
                    await page.wait_for_selector(RESULTS_TABLE_SELECTOR, state="visible", timeout=element_timeout_ws)

                    current_page_items_raw = await extract_table_data_from_current_list_page(
                        page, processed_list_pages, target_site_base_url, site_key, run_id, element_timeout_ws
                    )
                    
                    if current_page_items_raw and isinstance(current_page_items_raw[0], dict) and current_page_items_raw[0].get("status_signal") == "NO_RECORDS_FOUND_IN_TABLE":
                        logger.info(f"'No records' signal on page {processed_list_pages}. Ending scrape.")
                        update_worker_status(run_specific_temp_dir, "FINISHED_NO_DATA_ON_PAGE")
                        was_successful_overall_scrape = True; break
                    
                    if not current_page_items_raw:
                        logger.info(f"No items on list page {processed_list_pages}.")
                        consecutive_empty_list_pages += 1
                        if consecutive_empty_list_pages >= 2: 
                            logger.warning("Two consecutive empty list pages. Ending scrape.")
                            update_worker_status(run_specific_temp_dir, "FINISHED_EMPTY_PAGES")
                            was_successful_overall_scrape = True; break
                    else:
                        consecutive_empty_list_pages = 0; detail_tasks = []
                        for item_data_list_page in current_page_items_raw:
                            if not isinstance(item_data_list_page, dict): continue
                            detail_url_from_list = item_data_list_page.get("view_status_link_full_url")
                            tender_id_from_list = item_data_list_page.get("tender_id", f"Item_P{processed_list_pages}_R{len(detail_tasks)+1}")
                            if detail_url_from_list and detail_url_from_list != "N/A":
                                task = process_single_tender_detail_page(
                                    detail_semaphore, context, detail_url_from_list, site_key, 
                                    tender_id_from_list, processed_list_pages, run_id,
                                    detail_page_timeout_ws, element_timeout_ws, popup_page_timeout_ws 
                                )
                                detail_tasks.append((task, item_data_list_page))
                            else:
                                augmented_item = item_data_list_page.copy(); augmented_item['summary_file_status'] = "no_detail_link"
                                tagged_block = _format_rot_tags(augmented_item, site_key);
                                if tagged_block: outfile_merged.write(tagged_block + "\n\n")
                        
                        if detail_tasks:
                            gathered_detail_results = await asyncio.gather(*(t[0] for t in detail_tasks), return_exceptions=True)
                            for i, res_or_exc in enumerate(gathered_detail_results):
                                original_item_data = detail_tasks[i][1]; augmented_item_data = original_item_data.copy()
                                status, rel_path, fname = ("error_gather", None, None) # Default
                                if isinstance(res_or_exc, Exception): status = f"error_gather_{type(res_or_exc).__name__}"
                                elif isinstance(res_or_exc, tuple) and len(res_or_exc) == 3: status, rel_path, fname = res_or_exc
                                else: status = "unknown_detail_err_type"
                                
                                augmented_item_data.update({'summary_file_status': status, 'summary_file_path_relative': rel_path or "N/A", 'summary_file_name': fname or "N/A"})
                                if status == "downloaded": total_tenders_processed_successfully +=1
                                tagged_block = _format_rot_tags(augmented_item_data, site_key)
                                if tagged_block: outfile_merged.write(tagged_block + "\n\n")
                    
                    next_page_locator = page.locator(PAGINATION_LINK_SELECTOR)
                    if await next_page_locator.is_visible(timeout=5000):
                        if processed_list_pages >= max_list_pages_ws: 
                            logger.info(f"Max list pages ({max_list_pages_ws}) reached."); 
                            update_worker_status(run_specific_temp_dir, "FINISHED_MAX_PAGES_REACHED"); 
                            was_successful_overall_scrape = True; break
                        await next_page_locator.click(); await page.wait_for_timeout(pagination_wait_ws)
                    else: 
                        logger.info("No 'Next' link. End of results."); 
                        update_worker_status(run_specific_temp_dir, "FINISHED_END_OF_RESULTS"); 
                        was_successful_overall_scrape = True; break
            
            status_file_path_check = run_specific_temp_dir / "status.txt"
            current_status_on_file = status_file_path_check.read_text(encoding='utf-8').strip() if status_file_path_check.is_file() else ""
            if not current_status_on_file.startswith(("FINISHED_", "ERROR_")): 
                update_worker_status(run_specific_temp_dir, "FINISHED_SUCCESS")
            if status_file_path_check.is_file() and status_file_path_check.read_text(encoding='utf-8').strip().startswith("FINISHED_"): 
                was_successful_overall_scrape = True

            logger.info(f"ROT scrape finished. Success items: {total_tenders_processed_successfully}. Merged data: {final_site_tagged_output_path}")

        except Exception as e:
            logger.critical(f"Major error in ROT worker: {e}", exc_info=True)
            update_worker_status(run_specific_temp_dir, f"ERROR_UNHANDLED_{type(e).__name__}")
            if page and not page.is_closed(): await page.screenshot(path=orchestration_debug_screenshots_dir / "debug_worker_critical_err.png")
            was_successful_overall_scrape = False
        finally:
            if context: 
                try: await context.close()
                except Exception as e_ctx: logger.error(f"Ctx close err: {e_ctx}")
            if browser: 
                try: await browser.close()
                except Exception as e_bwsr: logger.error(f"Browser close err: {e_bwsr}")
            logger.info("Playwright resources closed.")
            
            final_status_check_msg = (run_specific_temp_dir / "status.txt").read_text(encoding='utf-8').strip() if (run_specific_temp_dir / "status.txt").is_file() else "UNKNOWN_FINAL"
            logger.info(f"Worker for {site_key} (Run: {run_id}) terminated with status: {final_status_check_msg}")

            if was_successful_overall_scrape:
                dashboard_host_port_env = os.environ.get("TENFIN_DASHBOARD_HOST_PORT", "localhost:8081") 
                if dashboard_host_port_env == "localhost:8081" and os.environ.get("TENFIN_DASHBOARD_HOST_PORT") is None:
                     logger.warning(f"TENFIN_DASHBOARD_HOST_PORT env var not found by worker; used default '{dashboard_host_port_env}'. Ensure dashboard passes this correctly if not default.")
                
                dashboard_consolidation_url = f"http://{dashboard_host_port_env}/trigger-rot-consolidation-from-worker"
                logger.info(f"Attempting to trigger ROT consolidation at: {dashboard_consolidation_url}")
                try:
                    async with httpx.AsyncClient(timeout=20.0) as client: 
                        response = await client.post(dashboard_consolidation_url)
                        if response.status_code == 200: logger.info(f"Consolidation triggered: {response.text}")
                        else: logger.error(f"Consolidation trigger failed: {response.status_code} - {response.text[:500]}")
                except httpx.RequestError as exc_http: logger.error(f"Consolidation HTTP err: {exc_http}")
                except Exception as e_trigger: logger.error(f"Consolidation unexpected err: {e_trigger}", exc_info=True)
            else:
                logger.info(f"Skipping ROT consolidation trigger; run not marked successful (Final Status: {final_status_check_msg}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless ROT Worker with interactive CAPTCHA & JSON extraction.") # Updated desc
    parser.add_argument("--site_key", required=True, help="Site key (e.g., CPPP, Assam) from settings.json.")
    parser.add_argument("--run_id", required=True, help="Unique ID for this worker run, provided by dashboard.")
    parser.add_argument("--site_url", required=True, help="Full URL of the ROT search page for the site.")
    parser.add_argument("--base_url", required=True, help="Base URL for resolving relative links for the site.")
    parser.add_argument("--tender_status", required=True, help="Tender status value to select.")
    parser.add_argument("--settings_json", type=str, help="JSON string of worker-specific settings (max_pages, timeouts etc.)")
    args = parser.parse_args()
    
    worker_run_settings = { 
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
            # Use a temporary simple print here as logger might not be set up yet if run directly with bad JSON
            print(f"WARNING: Could not parse --settings_json: '{args.settings_json}'. Using defaults.")

    # Initial print statements for direct execution feedback
    print(f"--- Headless ROT Worker Initializing (with JSON extraction call) ---")
    print(f"Site Key: {args.site_key}, Run ID: {args.run_id}, Tender Status: {args.tender_status}")
    print(f"Using Search URL: {args.site_url}, Base URL: {args.base_url}")
    print(f"Effective Worker Settings: {worker_run_settings}")
    print(f"Communication Dir: {TEMP_WORKER_RUNS_DIR / args.run_id}")
    print(f"HTML Output Base: {ROT_DETAIL_HTMLS_DIR / _clean_path_component_worker(args.site_key)}") # Use cleaner for path
    print(f"Tagged Data Output (Append): {ROT_MERGED_SITE_SPECIFIC_DIR / f'Merged_ROT_{clean_filename_rot(args.site_key)}_{TODAY_STR}.txt'}")
    print(f"Structured JSON Output Base: {ROT_STRUCTURED_JSON_DIR / _clean_path_component_worker(args.site_key)}")
    print("-" * 70)

    # Ensure base directories are created
    TEMP_WORKER_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ROT_DETAIL_HTMLS_DIR.mkdir(parents=True, exist_ok=True)
    ROT_MERGED_SITE_SPECIFIC_DIR.mkdir(parents=True, exist_ok=True)
    ROT_STRUCTURED_JSON_DIR.mkdir(parents=True, exist_ok=True)
    
    asyncio.run(run_headless_rot_scrape_orchestration(
        args.site_key, args.run_id, args.site_url, args.base_url, args.tender_status, worker_run_settings
    ))
    print(f"--- Headless ROT Worker for {args.site_key} (Run ID: {args.run_id}) Finished ---")
