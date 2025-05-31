#!/usr/bin/env python3
# File: headless_rot_worker.py (Adapted from your stand.py for interactive CAPTCHA)

import asyncio
import base64
import argparse
import datetime
from pathlib import Path
import re
import logging
import json
from urllib.parse import urljoin, urlparse
from typing import Tuple, Optional, Dict, Set, List, Any # Ensure Any is imported

from playwright.async_api import async_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup, Tag

# --- Worker Configuration ---
# These will be passed as arguments or derived
# LIVE_SITE_ROT_SEARCH_URL = "https://eprocure.gov.in/eprocure/app?page=WebTenderStatusLists&service=page" # Now from args
# LIVE_SITE_BASE_URL = "https://eprocure.gov.in/eprocure/app/" # Now from args
# DEFAULT_TENDER_STATUS = "5" # Now from args

TENDER_STATUS_DROPDOWN_SELECTOR = "#tenderStatus"
CAPTCHA_IMAGE_SELECTOR = "#captchaImage"
CAPTCHA_TEXT_INPUT_SELECTOR = "#captchaText"
SEARCH_BUTTON_SELECTOR = "#Search"
RESULTS_TABLE_SELECTOR = "#tabList"
VIEW_STATUS_LINK_IN_RESULTS = "table#tabList tr[id^='informal'] a[id^='view']"
SUMMARY_LINK_ON_DETAIL_PAGE = "a#DirectLink_0:has-text('Click link to view the all stage summary Details')"
ERROR_MESSAGE_SELECTORS = ".error_message, .errormsg, #msgDiv"
PAGINATION_LINK_SELECTOR = "a#loadNext"

# --- Default Settings (can be overridden by global settings if worker loads them, but simpler to pass) ---
MAX_ROT_LIST_PAGES_TO_FETCH_DEFAULT = 3
DETAIL_PROCESSING_CONCURRENCY_DEFAULT = 2
WAIT_AFTER_PAGINATION_CLICK_MS_DEFAULT = 7000
PAGE_LOAD_TIMEOUT_DEFAULT = 60000
DETAIL_PAGE_TIMEOUT_DEFAULT = 75000
POPUP_PAGE_TIMEOUT_DEFAULT = 60000
ELEMENT_TIMEOUT_DEFAULT = 20000
POST_SUBMIT_TIMEOUT_DEFAULT = 30000
CAPTCHA_ANSWER_WAIT_TIMEOUT_S = 300 # 5 minutes to solve CAPTCHA

# --- Paths ---
try:
    SCRIPT_DIR_WORKER = Path(__file__).parent.resolve()
except NameError:
    SCRIPT_DIR_WORKER = Path('.').resolve()

# Communication directory with the dashboard (must match dashboard's constant)
TEMP_RUN_DATA_DIR_BASE = SCRIPT_DIR_WORKER / "temp_run_data"

# Final output directories for scraped data
FINAL_DATA_ROOT = SCRIPT_DIR_WORKER / "scraped_data_rot"
FINAL_DOWNLOADS_DIR_BASE = FINAL_DATA_ROOT / "DetailDownloads"
FINAL_MERGED_DIR_BASE = FINAL_DATA_ROOT / "SiteSpecificMergedROT"

SCRIPT_NAME_TAG = Path(__file__).stem
TODAY_STR = datetime.date.today().strftime("%Y-%m-%d")


# --- Logging Setup (Worker Specific) ---
logger = logging.getLogger(f"{SCRIPT_NAME_TAG}_worker") # Default logger name
# Logging will be configured more specifically in main based on run_id/site_key

def setup_worker_logging(site_key_for_log: str, run_id_for_log: str):
    global logger
    logger_name = f"{SCRIPT_NAME_TAG}_{site_key_for_log}_{run_id_for_log}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    log_dir_rot_worker = SCRIPT_DIR_WORKER / "logs_rot" / site_key_for_log
    log_dir_rot_worker.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir_rot_worker / f"worker_{run_id_for_log}_{TODAY_STR}.log"

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
    popup_page: Page, site_key: str, tender_file_id: str
) -> Tuple[str, Optional[str], Optional[str]]: # status, relative_path, filename
    """Processes the final popup, saves HTML, returns status and file info."""
    
    tender_output_dir_htmls = FINAL_DOWNLOADS_DIR_BASE / site_key / clean_filename_rot(tender_file_id)
    tender_output_dir_htmls.mkdir(parents=True, exist_ok=True)
    
    log_prefix_popup = f"[{clean_filename_rot(tender_file_id)} Popup]"
    logger.info(f"{log_prefix_popup} Processing final popup page: {popup_page.url}")
    
    try:
        await popup_page.wait_for_load_state("domcontentloaded", timeout=POPUP_PAGE_TIMEOUT_DEFAULT)
        html_content_report = await popup_page.content()

        if html_content_report:
            processed_html_content = _remove_html_protection(html_content_report)
            safe_id_part = clean_filename_rot(tender_file_id) # tender_file_id is usually the tender_id
            save_filename_html = f"ROT_{safe_id_part}_StageSummary.html"
            save_path_html_absolute = tender_output_dir_htmls / save_filename_html
            
            save_path_html_absolute.write_text(processed_html_content, encoding='utf-8', errors='replace')
            
            # Make path relative to FINAL_DOWNLOADS_DIR_BASE for storage in merged file
            try:
                save_path_html_relative = save_path_html_absolute.relative_to(FINAL_DOWNLOADS_DIR_BASE)
            except ValueError: # If they are not on the same drive or path structure is unexpected
                logger.warning(f"{log_prefix_popup} Could not make path relative for {save_path_html_absolute}. Storing absolute.")
                save_path_html_relative = save_path_html_absolute
            
            logger.info(f"{log_prefix_popup} ✅ SAVED HTML: {save_path_html_absolute.name}")
            return "downloaded", str(save_path_html_relative), save_filename_html
        else:
            logger.warning(f"{log_prefix_popup} No HTML content from final popup page.")
            return "download_no_content", None, None
    except PlaywrightTimeout:
        logger.error(f"{log_prefix_popup} Timeout waiting for popup content.", exc_info=False)
        await popup_page.screenshot(path=tender_output_dir_htmls / f"DEBUG_POPUP_TIMEOUT_{SCRIPT_NAME_TAG}.png")
        return "download_timeout", None, None
    except Exception as e:
        logger.error(f"{log_prefix_popup} Error saving popup HTML: {e}", exc_info=True)
        await popup_page.screenshot(path=tender_output_dir_htmls / f"DEBUG_POPUP_ERROR_{SCRIPT_NAME_TAG}.png")
        return f"download_error_{type(e).__name__}", None, None


async def process_single_tender_detail_page( # Returns status, relative_path, filename
    semaphore: asyncio.Semaphore, main_browser_context: BrowserContext,
    detail_page_url: str, site_key: str,
    tender_display_id: str, list_page_num: int
) -> Tuple[str, Optional[str], Optional[str]]:
    
    log_prefix = f"[ListPage{list_page_num}_{clean_filename_rot(tender_display_id)}]"
    logger.info(f"{log_prefix} Starting detail processing for URL: {detail_page_url}")
    
    detail_page: Optional[Page] = None
    popup_page: Optional[Page] = None
    
    # Output directory for debug screenshots related to this specific tender's detail processing
    debug_output_dir_for_tender = FINAL_DOWNLOADS_DIR_BASE / site_key / clean_filename_rot(tender_display_id) / "debug"
    debug_output_dir_for_tender.mkdir(parents=True, exist_ok=True)

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
            
            # _handle_final_popup_content now takes site_key and tender_display_id
            # and saves to the correct FINAL_DOWNLOADS_DIR_BASE structure
            status, rel_path, fname = await _handle_final_popup_content(
                popup_page, site_key, tender_display_id 
            )
            
            if status == "downloaded":
                logger.info(f"{log_prefix} Successfully processed detail and popup. File: {fname}")
            else:
                logger.warning(f"{log_prefix} Failed to process content from popup. Status: {status}")
            return status, rel_path, fname

        except PlaywrightTimeout as pt_err:
            logger.error(f"{log_prefix} Timeout in detail/popup processing for '{detail_page_url}': {pt_err}", exc_info=False)
            if detail_page and not detail_page.is_closed():
                await detail_page.screenshot(path=debug_output_dir_for_tender / f"DETAIL_PG_TIMEOUT.png")
            if popup_page and not popup_page.is_closed():
                await popup_page.screenshot(path=debug_output_dir_for_tender / f"POPUP_PG_TIMEOUT.png")
            return f"detail_processing_timeout", None, None
        except Exception as e:
            logger.error(f"{log_prefix} Error processing detail link '{detail_page_url}': {e}", exc_info=True)
            return f"detail_processing_error_{type(e).__name__}", None, None
        finally:
            if popup_page and not popup_page.is_closed(): await popup_page.close()
            if detail_page and not detail_page.is_closed(): await detail_page.close()

async def extract_table_data_from_current_list_page(page: Page, list_page_num: int, site_base_url_for_links: str) -> List[Dict[str, Any]]:
    extracted_data: list[dict[str, Any]] = []
    logger.info(f"[ListPage{list_page_num}] Extracting table data...")
    # Debug screenshot directory for table extraction issues
    debug_table_extract_dir = TEMP_RUN_DATA_DIR_BASE / page.context.browser.contexts[0].pages[0].url.split("run_id=")[-1] / "debug_table_extracts" # Quick hack to get run_id from a page url if possible, needs refinement
    if "run_id=" not in page.context.browser.contexts[0].pages[0].url : # Fallback if run_id not in URL
        debug_table_extract_dir = SCRIPT_DIR_WORKER / "debug_table_extracts" # Generic fallback
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
                    "view_status_link_full_url": full_detail_url # Added for direct use
                })
            else: logger.warning(f"[ListPage{list_page_num}] Row {row_idx+1} has {len(cols)} cols, expected 6. Skipping.")
        logger.info(f"[ListPage{list_page_num}] Extracted {len(extracted_data)} items.")
    except PlaywrightTimeout:
        logger.error(f"[ListPage{list_page_num}] Timeout getting HTML of '{RESULTS_TABLE_SELECTOR}'.")
        await page.screenshot(path=debug_table_extract_dir / f"TABLE_EXTRACT_TIMEOUT_P{list_page_num}_{SCRIPT_NAME_TAG}.png")
    except Exception as e:
        logger.error(f"[ListPage{list_page_num}] Error extracting table data: {e}", exc_info=True)
        await page.screenshot(path=debug_table_extract_dir / f"TABLE_EXTRACT_ERROR_P{list_page_num}_{SCRIPT_NAME_TAG}.png")
    return extracted_data


async def run_headless_rot_scrape_orchestration(
    site_key: str, run_id: str,
    target_site_search_url: str, target_site_base_url: str, tender_status_to_select: str,
    worker_settings: Dict[str, Any] # To pass max_pages, concurrency, timeouts
):
    run_specific_temp_dir = TEMP_RUN_DATA_DIR_BASE / run_id
    run_specific_temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging for this specific run
    setup_worker_logging(site_key, run_id)
    logger.info(f"--- Starting Headless ROT Worker for Site: {site_key}, Run ID: {run_id} ---")
    logger.info(f"Target Search URL: {target_site_search_url}")
    logger.info(f"Target Base URL for links: {target_site_base_url}")
    logger.info(f"Tender Status to Select: {tender_status_to_select}")
    logger.info(f"Worker Settings: {worker_settings}")

    # Prepare final output file path (appended to by worker)
    FINAL_MERGED_DIR_BASE.mkdir(parents=True, exist_ok=True)
    final_site_tagged_output_path = FINAL_MERGED_DIR_BASE / f"Merged_ROT_{clean_filename_rot(site_key)}_{TODAY_STR}.txt"

    update_worker_status(run_specific_temp_dir, "WORKER_STARTED")

    async with async_playwright() as playwright:
        browser = None
        context = None
        try:
            browser = await playwright.chromium.launch(headless=True) # Consider non-headless for debugging
            context = await browser.new_context(ignore_https_errors=True, accept_downloads=True)
            page = await context.new_page()
            
            update_worker_status(run_specific_temp_dir, "FETCHING_CAPTCHA")
            logger.info(f"Navigating to: {target_site_search_url}")
            await page.goto(target_site_search_url, wait_until="domcontentloaded", timeout=worker_settings.get("page_load_timeout", PAGE_LOAD_TIMEOUT_DEFAULT))

            if not await _save_captcha_for_dashboard(page, run_specific_temp_dir, tender_status_to_select):
                logger.error("Failed to save CAPTCHA for dashboard. Aborting worker.")
                # Status already set by _save_captcha_for_dashboard on error
                return

            captcha_solution = await _wait_for_captcha_solution(run_specific_temp_dir)
            if not captcha_solution:
                logger.error("No CAPTCHA solution received or timeout. Aborting worker.")
                # Status already set by _wait_for_captcha_solution on error/timeout
                return
            
            update_worker_status(run_specific_temp_dir, "PROCESSING_WITH_CAPTCHA")
            logger.info(f"CAPTCHA solution ('******') received. Submitting form...")
            await page.locator(CAPTCHA_TEXT_INPUT_SELECTOR).fill(captcha_solution)
            await page.locator(SEARCH_BUTTON_SELECTOR).click()
            
            logger.info(f"Waiting for initial results ('{RESULTS_TABLE_SELECTOR}') OR error.")
            try:
                await page.wait_for_selector(f"{RESULTS_TABLE_SELECTOR}, {ERROR_MESSAGE_SELECTORS}", state="visible", timeout=worker_settings.get("post_submit_timeout", POST_SUBMIT_TIMEOUT_DEFAULT))
            except PlaywrightTimeout:
                logger.error("Timeout: Neither results table nor error message appeared after CAPTCHA submission.")
                await page.screenshot(path=run_specific_temp_dir / f"debug_captcha_submit_overall_timeout.png")
                update_worker_status(run_specific_temp_dir, "ERROR_CAPTCHA_SUBMIT_TIMEOUT")
                return

            if not await page.locator(RESULTS_TABLE_SELECTOR).is_visible(timeout=2000): # Quick check
                err_text = "Unknown (Results table not found)"
                err_loc = page.locator(ERROR_MESSAGE_SELECTORS).first
                if await err_loc.count() > 0 and await err_loc.is_visible(timeout=1000): # Check if error element exists and is visible
                    err_text_content = await err_loc.text_content()
                    err_text = err_text_content.strip() if err_text_content else err_text
                logger.error(f"CAPTCHA submission failed or error on page: {err_text}")
                await page.screenshot(path=run_specific_temp_dir / f"debug_captcha_submit_error.png")
                update_worker_status(run_specific_temp_dir, f"ERROR_CAPTCHA_SUBMIT_{clean_filename_rot(err_text[:50])}")
                return
            
            logger.info("✅ Initial results table detected after CAPTCHA. Proceeding with scrape.")
            update_worker_status(run_specific_temp_dir, "SCRAPING_RESULTS")
            
            processed_list_pages = 0
            detail_semaphore = asyncio.Semaphore(worker_settings.get("detail_concurrency", DETAIL_PROCESSING_CONCURRENCY_DEFAULT))
            consecutive_empty_list_pages = 0
            max_list_pages = worker_settings.get("max_list_pages", MAX_ROT_LIST_PAGES_TO_FETCH_DEFAULT)
            total_tenders_processed_successfully = 0

            with open(final_site_tagged_output_path, "a", encoding="utf-8") as outfile_merged: # Open in append mode
                while processed_list_pages < max_list_pages:
                    processed_list_pages += 1
                    logger.info(f"--- Processing List Page: {processed_list_pages} ---")
                    update_worker_status(run_specific_temp_dir, f"PROCESSING_LIST_PAGE_{processed_list_pages}")
                    await page.wait_for_selector(RESULTS_TABLE_SELECTOR, state="visible", timeout=worker_settings.get("element_timeout", ELEMENT_TIMEOUT_DEFAULT))
                    
                    current_page_items_raw = await extract_table_data_from_current_list_page(page, processed_list_pages, target_site_base_url)
                    
                    if current_page_items_raw and isinstance(current_page_items_raw[0], dict) and current_page_items_raw[0].get("status_signal") == "NO_RECORDS_FOUND_IN_TABLE":
                        logger.info(f"'No records' signal on page {processed_list_pages}. Ending scrape for this site.")
                        update_worker_status(run_specific_temp_dir, "FINISHED_NO_DATA_ON_PAGE")
                        break
                    
                    if not current_page_items_raw:
                        logger.info(f"No items found on list page {processed_list_pages}.")
                        consecutive_empty_list_pages += 1
                        if consecutive_empty_list_pages >= 2:
                            logger.warning("Two consecutive empty list pages. Ending scrape for this site.")
                            update_worker_status(run_specific_temp_dir, "FINISHED_EMPTY_PAGES")
                            break
                    else:
                        consecutive_empty_list_pages = 0
                        
                        detail_tasks = []
                        for item_data_list_page in current_page_items_raw:
                            # Ensure it's a dict before proceeding (it should be)
                            if not isinstance(item_data_list_page, dict): 
                                logger.warning(f"Skipping non-dict item from list page: {item_data_list_page}")
                                continue

                            detail_url_from_list = item_data_list_page.get("view_status_link_full_url")
                            tender_id_from_list = item_data_list_page.get("tender_id", f"Item_P{processed_list_pages}_R{len(detail_tasks)+1}")

                            if detail_url_from_list and detail_url_from_list != "N/A":
                                task = process_single_tender_detail_page(
                                    detail_semaphore, context, detail_url_from_list,
                                    site_key, tender_id_from_list, processed_list_pages
                                )
                                detail_tasks.append((task, item_data_list_page)) # Pair task with its original list data
                            else: # No detail link, just format what we have from the list
                                augmented_item = item_data_list_page.copy()
                                augmented_item['summary_file_status'] = "no_detail_link"
                                tagged_block = _format_rot_tags(augmented_item, site_key)
                                if tagged_block: outfile_merged.write(tagged_block + "\n\n")
                        
                        if detail_tasks:
                            logger.info(f"Gathering {len(detail_tasks)} detail processing tasks for list page {processed_list_pages}...")
                            # Results will be tuples of (status, rel_path, fname)
                            gathered_detail_results = await asyncio.gather(*(task_tuple[0] for task_tuple in detail_tasks))
                            logger.info(f"All detail tasks for list page {processed_list_pages} finished.")

                            for i, detail_result_tuple in enumerate(gathered_detail_results):
                                original_list_item_data = detail_tasks[i][1] # Get corresponding list item
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
                    if await next_page_locator.is_visible(timeout=5000): # Quick check for next link
                        if processed_list_pages >= max_list_pages:
                            logger.info(f"Reached max list pages ({max_list_pages}). Stopping.")
                            update_worker_status(run_specific_temp_dir, "FINISHED_MAX_PAGES_REACHED")
                            break
                        logger.info(f"Clicking 'Next' for list page {processed_list_pages + 1}.")
                        await next_page_locator.click()
                        logger.info(f"Waiting {worker_settings.get('pagination_wait', WAIT_AFTER_PAGINATION_CLICK_MS_DEFAULT) / 1000}s for next page results...")
                        await page.wait_for_timeout(worker_settings.get('pagination_wait', WAIT_AFTER_PAGINATION_CLICK_MS_DEFAULT))
                    else:
                        logger.info(f"No 'Next' page link found after list page {processed_list_pages}. End of results.")
                        update_worker_status(run_specific_temp_dir, "FINISHED_END_OF_RESULTS")
                        break
            
            if not (processed_list_pages < max_list_pages and consecutive_empty_list_pages < 2): # If loop exited by break conditions
                 pass # Status already set
            else: # Loop finished due to max_list_pages or normal completion
                update_worker_status(run_specific_temp_dir, "FINISHED_SUCCESS")

            logger.info(f"ROT scrape processing finished for site {site_key}. Successfully processed details for {total_tenders_processed_successfully} tenders.")
            logger.info(f"Final merged ROT data for site {site_key} is in: {final_site_tagged_output_path}")

        except Exception as e:
            logger.critical(f"Major error in ROT worker for site {site_key}: {e}", exc_info=True)
            update_worker_status(run_specific_temp_dir, f"ERROR_UNHANDLED_{type(e).__name__}")
            # Save debug screenshot of current page if error occurs
            if page and not page.is_closed():
                await page.screenshot(path=run_specific_temp_dir / "debug_worker_critical_error.png")

        finally:
            if context: await context.close()
            if browser: await browser.close()
            logger.info(f"Playwright resources for ROT worker (Site: {site_key}, Run: {run_id}) closed.")
            # Log final status one last time, could be redundant if already set correctly
            final_status_check = (run_specific_temp_dir / "status.txt").read_text(encoding='utf-8').strip() if (run_specific_temp_dir / "status.txt").is_file() else "UNKNOWN_FINAL"
            logger.info(f"Worker for {site_key} (Run: {run_id}) terminated with status: {final_status_check}")

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
    print(f"Communication Dir: {TEMP_RUN_DATA_DIR_BASE / args.run_id}")
    print(f"Final HTML Output Base: {FINAL_DOWNLOADS_DIR_BASE / args.site_key}")
    print(f"Final Tagged Data Output File (Append Mode): {FINAL_MERGED_DIR_BASE / f'Merged_ROT_{clean_filename_rot(args.site_key)}_{TODAY_STR}.txt'}")
    print("-" * 70)

    # Create base directories if they don't exist
    TEMP_RUN_DATA_DIR_BASE.mkdir(parents=True, exist_ok=True)
    FINAL_DOWNLOADS_DIR_BASE.mkdir(parents=True, exist_ok=True)
    FINAL_MERGED_DIR_BASE.mkdir(parents=True, exist_ok=True)
    
    asyncio.run(run_headless_rot_scrape_orchestration(
        args.site_key,
        args.run_id,
        args.site_url,
        args.base_url,
        args.tender_status,
        worker_run_settings
    ))
    print(f"--- Headless ROT Worker for {args.site_key} (Run ID: {args.run_id}) Finished ---")
