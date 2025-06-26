#!/usr/bin/env python3
# File: headless_rot_worker.py

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
import sys

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
VIEW_STATUS_LINK_IN_RESULTS = "table#tabList tr[id^='informal'] a[id^='view']"
SUMMARY_LINK_ON_DETAIL_PAGE = "a#DirectLink_0:has-text('Click link to view the all stage summary Details')"
ERROR_MESSAGE_SELECTORS = ".error_message, .errormsg, #msgDiv"
PAGINATION_LINK_SELECTOR = "a#loadNext"

# --- Default Settings ---
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
except NameError:
    PROJECT_ROOT = Path('.').resolve()

SITE_DATA_ROOT = PROJECT_ROOT / "site_data"
ROT_DATA_DIR = SITE_DATA_ROOT / "ROT"
ROT_MERGED_SITE_SPECIFIC_DIR = ROT_DATA_DIR / "MergedSiteSpecific"
ROT_DETAIL_HTMLS_DIR = ROT_DATA_DIR / "DetailHtmls"
TEMP_DATA_DIR = SITE_DATA_ROOT / "TEMP"
TEMP_WORKER_RUNS_DIR = TEMP_DATA_DIR / "WorkerRuns" 
TEMP_DEBUG_SCREENSHOTS_DIR = TEMP_DATA_DIR / "DebugScreenshots"
LOGS_BASE_DIR_WORKER = PROJECT_ROOT / "LOGS"
SCRIPT_NAME_TAG = Path(__file__).stem
TODAY_STR = datetime.date.today().strftime("%Y-%m-%d")

# --- Logging Setup ---
logger = logging.getLogger(f"{SCRIPT_NAME_TAG}_placeholder")

def setup_worker_logging(site_key_for_log: str, run_id_for_log: str):
    global logger 
    # Sanitize site_key for directory creation
    safe_site_key_for_dir = _clean_path_component_worker(site_key_for_log)
    
    logger_name = f"{SCRIPT_NAME_TAG}_worker_{safe_site_key_for_dir}_{run_id_for_log[-6:]}"
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers(): logger.handlers.clear(); logger.propagate = False 
    logger.setLevel(logging.INFO); logger.propagate = False 
    
    log_dir_for_this_worker = LOGS_BASE_DIR_WORKER / "rot_worker" / safe_site_key_for_dir
    log_dir_for_this_worker.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir_for_this_worker / f"worker_{run_id_for_log}_{TODAY_STR}.log"
    
    fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter(f"%(asctime)s [%(levelname)s] (ROTWorker-{site_key_for_log} Run-{run_id_for_log[-6:]}) %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(f"[ROTWorker-{site_key_for_log} Run-{run_id_for_log[-6:]}] %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    
    logger.info(f"Worker logging initialized. Log file: {log_file_path}")

# --- Utility Functions ---
def _remove_html_protection(html_content: str) -> str:
    modified_content = re.sub(r'<script language="javascript" type="text/javascript">\s*if\s*\(\s*window\.name\s*==\s*""\s*\)\s*\{[^}]*window\.location\.replace[^}]*\}[^<]*else if\s*\(\s*window\.name\.indexOf\("Popup"\)\s*==\s*-1\)\s*\{[^}]*window\.location\.replace[^}]*\}\s*</script>', "<!-- Protection Script Removed -->", html_content, flags=re.IGNORECASE | re.DOTALL)
    modified_content = re.sub(r'<noscript>\s*<meta http-equiv="Refresh"[^>]*?UnauthorisedAccessPage[^>]*?>\s*</noscript>', "<!-- Noscript Redirect Removed -->", modified_content, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r'<body[^>]*>', lambda m: re.sub(r'\s*onpageshow\s*=\s*".*?"', '', m.group(0), flags=re.IGNORECASE), modified_content, count=1, flags=re.IGNORECASE)

def clean_filename_rot(filename: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|\r\n\t]+', '_', filename)
    return re.sub(r'[\s_]+', '_', cleaned).strip('_')[:150]


def update_worker_status(run_dir: Path, status_message: str):
    try:
        (run_dir / "status.txt").write_text(status_message, encoding='utf-8')
        if hasattr(logger, 'info'): logger.info(f"Worker status updated to: {status_message}")
    except Exception as e:
        if hasattr(logger, 'error'): logger.error(f"Error writing worker status '{status_message}': {e}")

def _clean_path_component_worker(component: str) -> str:
    if not isinstance(component, str): return ""
    cleaned = re.sub(r'[^\w.\-]', '', component)
    return "" if cleaned in [".", ".."] else cleaned

async def _save_captcha_for_dashboard(page: Page, run_dir: Path, tender_status_to_select: str, element_timeout: int) -> bool:
    captcha_b64_file = run_dir / "captcha.b64"
    try:
        logger.info(f"Selecting tender status '{tender_status_to_select}'.")
        await page.locator(TENDER_STATUS_DROPDOWN_SELECTOR).select_option(value=tender_status_to_select, timeout=element_timeout)
        await page.wait_for_timeout(3000) 
        captcha_element = page.locator(CAPTCHA_IMAGE_SELECTOR)
        await captcha_element.wait_for(state="visible", timeout=element_timeout)
        captcha_image_bytes = await captcha_element.screenshot(type='png')
        if not captcha_image_bytes: logger.error("CAPTCHA screenshot resulted in empty bytes."); return False
        captcha_b64_file.write_text(base64.b64encode(captcha_image_bytes).decode('utf-8'), encoding='utf-8')
        logger.info(f"CAPTCHA base64 data saved to: {captcha_b64_file}")
        update_worker_status(run_dir, "WAITING_CAPTCHA")
        return True
    except Exception as e:
        logger.error(f"Error saving CAPTCHA for dashboard: {e}", exc_info=True)
        update_worker_status(run_dir, f"ERROR_SAVING_CAPTCHA_{type(e).__name__}")
        return False

async def _wait_for_captcha_solution(run_dir: Path) -> Optional[str]:
    answer_file = run_dir / "answer.txt"
    logger.info(f"Waiting for CAPTCHA solution in {answer_file} (max {CAPTCHA_ANSWER_WAIT_TIMEOUT_S}s)...")
    for _ in range(CAPTCHA_ANSWER_WAIT_TIMEOUT_S): 
        if answer_file.is_file():
            try:
                solution = answer_file.read_text(encoding='utf-8').strip()
                if solution: logger.info("CAPTCHA solution received."); return solution
            except Exception as e:
                logger.error(f"Error reading answer file {answer_file}: {e}"); return None 
        await asyncio.sleep(1)
    logger.warning("Timeout waiting for CAPTCHA solution."); update_worker_status(run_dir, "TIMEOUT_CAPTCHA_INPUT"); return None

def _format_rot_tags(data_dict: Dict[str, Any], site_key: str) -> str:
    tags = ["--- TENDER START ---"]
    for key, val in data_dict.items():
        if val not in [None, "N/A"]:
            tag_name = ''.join(word.capitalize() for word in key.split('_'))
            tags.append(f"<{tag_name}>{str(val).strip()}</{tag_name}>")
    tags.append(f"<SourceSiteKey>{site_key}</SourceSiteKey>")
    tags.append("--- TENDER END ---")
    return "\n".join(tags)

async def _handle_final_popup_content(popup_page: Page, site_key: str, tender_file_id: str, run_id: str, popup_page_timeout: int) -> Tuple[str, Optional[str], Optional[str]]:
    cleaned_site_key_for_dir = _clean_path_component_worker(site_key)
    site_specific_html_output_dir = ROT_DETAIL_HTMLS_DIR / cleaned_site_key_for_dir
    site_specific_html_output_dir.mkdir(parents=True, exist_ok=True)
    log_prefix_popup = f"[{clean_filename_rot(tender_file_id)} Popup]"

    try:
        await popup_page.wait_for_load_state("domcontentloaded", timeout=popup_page_timeout)
        html_content_report_raw = await popup_page.content()

        if html_content_report_raw:
            processed_html_content = _remove_html_protection(html_content_report_raw)
            safe_id_part = clean_filename_rot(tender_file_id)
            timestamp_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename_html = f"ROT_{safe_id_part}_{timestamp_suffix}_StageSummary.html"
            save_path_html_absolute = site_specific_html_output_dir / save_filename_html
            save_path_html_absolute.write_text(processed_html_content, encoding='utf-8', errors='replace')
            
            logger.info(f"{log_prefix_popup} ✅ CLEANED & SAVED HTML: {save_filename_html}")

            path_to_extraction_script = PROJECT_ROOT / "utils" / "extract_structured_rot.py"
            if path_to_extraction_script.is_file():
                cmd = [
                    sys.executable,
                    str(path_to_extraction_script),
                    str(save_path_html_absolute),
                    "--site-key", site_key
                ]
                logger.info(f"{log_prefix_popup} Attempting to save structured data to DB: {' '.join(cmd)}")
                process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await process.communicate()
                if process.returncode == 0:
                    logger.info(f"{log_prefix_popup} Successfully processed structured data for {save_filename_html}.")
                    if stdout and stdout.strip(): logger.debug(f"{log_prefix_popup} Extractor STDOUT: {stdout.decode(errors='replace').strip()}")
                else:
                    logger.error(f"{log_prefix_popup} Failed to process structured data for {save_filename_html}. RC: {process.returncode}")
                    if stderr and stderr.strip(): logger.error(f"{log_prefix_popup} Extractor STDERR: {stderr.decode(errors='replace').strip()}")
            else:
                logger.warning(f"{log_prefix_popup} Data extraction script NOT FOUND at {path_to_extraction_script}. Skipping DB save.")
            
            return "downloaded", str(save_path_html_absolute.relative_to(PROJECT_ROOT)), save_filename_html
        else:
            return "download_no_content", None, None
    except Exception as e:
        logger.error(f"{log_prefix_popup} Error in popup handling: {e}", exc_info=True)
        return f"download_error_{type(e).__name__}", None, None

async def process_single_tender_detail_page(semaphore: asyncio.Semaphore, main_browser_context: BrowserContext, detail_page_url: str, site_key: str, tender_display_id: str, list_page_num: int, run_id: str, detail_page_timeout: int, element_timeout: int, popup_page_timeout_for_detail: int) -> Tuple[str, Optional[str], Optional[str]]:
    log_prefix = f"[ListPage{list_page_num}_{clean_filename_rot(tender_display_id)}]"
    detail_page: Optional[Page] = None; popup_page: Optional[Page] = None
    async with semaphore:
        try:
            detail_page = await main_browser_context.new_page()
            await detail_page.goto(detail_page_url, wait_until="domcontentloaded", timeout=detail_page_timeout)
            summary_link_locator = detail_page.locator(SUMMARY_LINK_ON_DETAIL_PAGE).first
            await summary_link_locator.wait_for(state="visible", timeout=element_timeout)
            logger.info(f"{log_prefix} Final summary link found. Clicking...")
            async with detail_page.expect_popup(timeout=popup_page_timeout_for_detail) as popup_info:
                await summary_link_locator.click()
            popup_page = await popup_info.value
            return await _handle_final_popup_content(popup_page, site_key, tender_display_id, run_id, popup_page_timeout_for_detail)
        except Exception as e:
            logger.error(f"{log_prefix} Error processing detail link '{detail_page_url}': {e}", exc_info=True)
            return f"detail_processing_error_{type(e).__name__}", None, None
        finally:
            if popup_page and not popup_page.is_closed(): await popup_page.close()
            if detail_page and not detail_page.is_closed(): await detail_page.close()

async def extract_table_data_from_current_list_page(page: Page, list_page_num: int, site_base_url_for_links: str) -> List[Dict[str, Any]]:
    extracted_data: List[Dict[str, Any]] = []
    logger.info(f"[ListPage{list_page_num}] Extracting table data...")
    try:
        results_table_html = await page.locator(RESULTS_TABLE_SELECTOR).inner_html(timeout=20000)
        soup = BeautifulSoup(results_table_html, "html.parser"); rows = soup.select("tr[id^='informal']")
        if not rows:
            if "no records found" in results_table_html.lower() or "no tenders available" in results_table_html.lower():
                return [{"status_signal": "NO_RECORDS_FOUND_IN_TABLE"}]
            logger.warning(f"[ListPage{list_page_num}] No data rows found."); return []
        for row_tag in rows:
            cols = row_tag.find_all("td")
            if len(cols) == 6:
                link_tag = cols[5].find("a", id=re.compile(r"^view_?\d*$"))
                href_val = link_tag['href'].strip() if link_tag and link_tag.has_attr('href') and link_tag['href'].strip() != '#' else "N/A"
                extracted_data.append({
                    "list_page_num": str(list_page_num), "rot_s_no": cols[0].get_text(strip=True), "rot_tender_id": cols[1].get_text(strip=True),
                    "rot_title_ref": cols[2].get_text(strip=True), "rot_organisation_chain": cols[3].get_text(strip=True),
                    "rot_tender_stage": cols[4].get_text(strip=True), "rot_status_detail_page_link": urljoin(site_base_url_for_links, href_val)
                })
        logger.info(f"[ListPage{list_page_num}] Extracted {len(extracted_data)} items.")
    except Exception as e:
        logger.error(f"[ListPage{list_page_num}] Error extracting table data: {e}", exc_info=True)
    return extracted_data

async def run_headless_rot_scrape_orchestration(
    site_key: str, run_id: str, 
    target_site_search_url: str, target_site_base_url: str, 
    tender_status_to_select: str, worker_settings: Dict[str, Any],
    from_date: Optional[str], to_date: Optional[str]
):
    run_specific_temp_dir = TEMP_WORKER_RUNS_DIR / run_id
    run_specific_temp_dir.mkdir(parents=True, exist_ok=True)
    setup_worker_logging(site_key, run_id) 
    logger.info(f"--- Starting Headless ROT Worker for Site: {site_key}, Run ID: {run_id} ---")
    logger.info(f"--- Dates: From={from_date}, To={to_date} ---")
    
    final_site_tagged_output_path = ROT_MERGED_SITE_SPECIFIC_DIR / f"Merged_ROT_{clean_filename_rot(site_key)}_{TODAY_STR}.txt"
    update_worker_status(run_specific_temp_dir, "WORKER_STARTED")
    
    page_load_timeout_ws = worker_settings.get("page_load_timeout", PAGE_LOAD_TIMEOUT_DEFAULT)
    detail_page_timeout_ws = worker_settings.get("detail_page_timeout", DETAIL_PAGE_TIMEOUT_DEFAULT)
    popup_page_timeout_ws = worker_settings.get("popup_page_timeout", POPUP_PAGE_TIMEOUT_DEFAULT)
    element_timeout_ws = worker_settings.get("element_timeout", ELEMENT_TIMEOUT_DEFAULT)
    post_submit_timeout_ws = worker_settings.get("post_submit_timeout", POST_SUBMIT_TIMEOUT_DEFAULT)
    pagination_wait_ws = worker_settings.get("pagination_wait", WAIT_AFTER_PAGINATION_CLICK_MS_DEFAULT)
    detail_concurrency_ws = worker_settings.get("detail_concurrency", DETAIL_PROCESSING_CONCURRENCY_DEFAULT)
    max_list_pages_ws = worker_settings.get("max_list_pages", MAX_ROT_LIST_PAGES_TO_FETCH_DEFAULT)
    
    async with async_playwright() as playwright:
        browser = None; context = None
        try:
            browser = await playwright.chromium.launch(headless=True, args=['--no-sandbox'])
            context = await browser.new_context(ignore_https_errors=True)
            page = await context.new_page()

            update_worker_status(run_specific_temp_dir, "FETCHING_CAPTCHA")
            await page.goto(target_site_search_url, wait_until="domcontentloaded", timeout=page_load_timeout_ws)
            if not await _save_captcha_for_dashboard(page, run_specific_temp_dir, tender_status_to_select, element_timeout_ws): return
            captcha_solution = await _wait_for_captcha_solution(run_specific_temp_dir)
            if not captcha_solution: return

            update_worker_status(run_specific_temp_dir, "PROCESSING_WITH_CAPTCHA")

            if from_date:
                formatted_from_date = datetime.datetime.strptime(from_date, '%Y-%m-%d').strftime('%d/%m/%Y')
                logger.info(f"Setting From Date to: {formatted_from_date} using JavaScript evaluation.")
                await page.evaluate(f"document.querySelector('#fromDate').value = '{formatted_from_date}';")

            if to_date:
                formatted_to_date = datetime.datetime.strptime(to_date, '%Y-%m-%d').strftime('%d/%m/%Y')
                logger.info(f"Setting To Date to: {formatted_to_date} using JavaScript evaluation.")
                await page.evaluate(f"document.querySelector('#toDate').value = '{formatted_to_date}';")

            await page.locator(CAPTCHA_TEXT_INPUT_SELECTOR).fill(captcha_solution)
            await page.locator(SEARCH_BUTTON_SELECTOR).click()
            
            await page.wait_for_selector(f"{RESULTS_TABLE_SELECTOR}, {ERROR_MESSAGE_SELECTORS}", state="visible", timeout=post_submit_timeout_ws)
            if not await page.locator(RESULTS_TABLE_SELECTOR).is_visible(timeout=2000):
                err_loc = page.locator(ERROR_MESSAGE_SELECTORS).first
                err_text = await err_loc.text_content() if await err_loc.count() > 0 else "Unknown error"
                logger.error(f"CAPTCHA submit failed: {err_text.strip()}"); update_worker_status(run_specific_temp_dir, f"ERROR_CAPTCHA_SUBMIT"); return

            update_worker_status(run_specific_temp_dir, "SCRAPING_RESULTS")
            processed_list_pages = 0; detail_semaphore = asyncio.Semaphore(detail_concurrency_ws)
            with open(final_site_tagged_output_path, "a", encoding="utf-8") as outfile_merged:
                while processed_list_pages < max_list_pages_ws:
                    processed_list_pages += 1
                    logger.info(f"--- Processing List Page: {processed_list_pages}/{max_list_pages_ws} ---")
                    current_page_items_raw = await extract_table_data_from_current_list_page(page, processed_list_pages, target_site_base_url)
                    if not current_page_items_raw: logger.info(f"No items on list page {processed_list_pages}."); break
                    if current_page_items_raw[0].get("status_signal") == "NO_RECORDS_FOUND_IN_TABLE": logger.info("'No records' signal received. Ending scrape."); break
                    
                    detail_tasks = []
                    for item_data in current_page_items_raw:
                        outfile_merged.write(_format_rot_tags(item_data, site_key) + "\n\n")
                        if item_data.get("rot_status_detail_page_link", "N/A") != "N/A":
                            detail_tasks.append(process_single_tender_detail_page(detail_semaphore, context, item_data["rot_status_detail_page_link"], site_key, item_data["rot_tender_id"], processed_list_pages, run_id, detail_page_timeout_ws, element_timeout_ws, popup_page_timeout_ws))
                    if detail_tasks: await asyncio.gather(*detail_tasks)

                    next_page_locator = page.locator(PAGINATION_LINK_SELECTOR)
                    if await next_page_locator.is_visible(timeout=5000):
                        await next_page_locator.click(); await page.wait_for_timeout(pagination_wait_ws)
                    else: logger.info("No 'Next' link. End of results."); break
            
            update_worker_status(run_specific_temp_dir, "FINISHED_SUCCESS")
        except Exception as e:
            logger.critical(f"Major error in ROT worker: {e}", exc_info=True)
            update_worker_status(run_specific_temp_dir, f"ERROR_UNHANDLED_{type(e).__name__}")
        finally:
            if context: await context.close()
            if browser: await browser.close()
            logger.info("Playwright resources closed.")
            final_status = (run_specific_temp_dir / "status.txt").read_text().strip() if (run_specific_temp_dir / "status.txt").exists() else "UNKNOWN"
            logger.info(f"Worker for {site_key} terminated with status: {final_status}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless ROT Worker with CAPTCHA & DB integration.")
    parser.add_argument("--site_key", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--site_url", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--tender_status", required=True)
    parser.add_argument("--settings_json", type=str)
    parser.add_argument("--from_date", type=str, help="From Date for search (YYYY-MM-DD)")
    parser.add_argument("--to_date", type=str, help="To Date for search (YYYY-MM-DD)")
    args = parser.parse_args()
    
    worker_run_settings = {}
    if args.settings_json:
        try: worker_run_settings.update(json.loads(args.settings_json))
        except json.JSONDecodeError: print(f"WARNING: Could not parse --settings_json.")
    
    for d in [TEMP_WORKER_RUNS_DIR, ROT_DETAIL_HTMLS_DIR, ROT_MERGED_SITE_SPECIFIC_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    asyncio.run(run_headless_rot_scrape_orchestration(
        args.site_key, args.run_id, args.site_url, args.base_url, args.tender_status, 
        worker_run_settings, args.from_date, args.to_date
    ))
