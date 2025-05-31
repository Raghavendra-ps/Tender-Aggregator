#!/usr/bin/env python3
# File: stand.py (Modified for headless operation with console CAPTCHA input)

import asyncio
import base64 
import argparse
from pathlib import Path
import re
import logging
import json 
from urllib.parse import urljoin, urlparse
# Ensure 'Any' is imported from typing
from typing import Tuple, Optional, Dict, Set, List, Any # <<<< CORRECTED IMPORT

from playwright.async_api import async_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup, Tag 

# --- Configuration (from your pasted script) ---
LIVE_SITE_ROT_SEARCH_URL = "https://eprocure.gov.in/eprocure/app?page=WebTenderStatusLists&service=page"
LIVE_SITE_BASE_URL = "https://eprocure.gov.in/eprocure/app/"
DEFAULT_TENDER_STATUS = "5" 

TENDER_STATUS_DROPDOWN_SELECTOR = "#tenderStatus" 
CAPTCHA_IMAGE_SELECTOR = "#captchaImage"        
CAPTCHA_TEXT_INPUT_SELECTOR = "#captchaText"    
SEARCH_BUTTON_SELECTOR = "#Search"              
RESULTS_TABLE_SELECTOR = "#tabList"             
VIEW_STATUS_LINK_IN_RESULTS = "table#tabList tr[id^='informal'] a[id^='view']" 
SUMMARY_LINK_ON_DETAIL_PAGE = "a#DirectLink_0:has-text('Click link to view the all stage summary Details')" 
ERROR_MESSAGE_SELECTORS = ".error_message, .errormsg, #msgDiv" 
PAGINATION_LINK_SELECTOR = "a#loadNext"        

MAX_ROT_LIST_PAGES_TO_FETCH = 3
DETAIL_PROCESSING_CONCURRENCY = 2 
WAIT_AFTER_PAGINATION_CLICK_MS = 7000

SCRIPT_DIR_STAND = Path(__file__).parent.resolve()
OUTPUT_DIR = SCRIPT_DIR_STAND / "stand_output_headless_captcha" 
CAPTCHA_IMAGE_SAVE_PATH = SCRIPT_DIR_STAND / "captcha_to_solve.png" 
SCRIPT_NAME_TAG = Path(__file__).stem
EXTRACTED_TABLE_DATA_FILENAME = "extracted_tender_list_data.json"

PAGE_LOAD_TIMEOUT = 60000
DETAIL_PAGE_TIMEOUT = 75000 
POPUP_PAGE_TIMEOUT = 60000    
DOWNLOAD_TIMEOUT_FOR_POPUP = 180000 
ELEMENT_TIMEOUT = 20000
POST_SUBMIT_TIMEOUT = 30000

logger = logging.getLogger("stand_headless_rot_logger")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    if logger.hasHandlers(): logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s'))
    logger.addHandler(ch)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# For this standalone script, these paths might not be directly used for final project structure,
# but they are used if _handle_final_popup_content tries to make paths relative to a common root.
BASE_DATA_DIR_ROT_ROOT_HEADLESS = SCRIPT_DIR_STAND / "scraped_data_rot_headless" 
DOWNLOADS_DIR_ROT_BASE_HEADLESS = BASE_DATA_DIR_ROT_ROOT_HEADLESS / "DetailDownloads"
SITE_SPECIFIC_MERGED_DIR_ROT_HEADLESS = BASE_DATA_DIR_ROT_ROOT_HEADLESS / "SiteSpecificMergedROT"
BASE_DATA_DIR_ROT_ROOT_HEADLESS.mkdir(parents=True, exist_ok=True)


# --- Utility Functions ---
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

# --- Core Scraping Functions ---
async def _save_captcha_and_get_solution(page: Page, captcha_image_path: Path, tender_status_to_select: str) -> Optional[str]:
    logger.info(f"Selecting tender status '{tender_status_to_select}'.")
    await page.locator(TENDER_STATUS_DROPDOWN_SELECTOR).select_option(value=tender_status_to_select, timeout=ELEMENT_TIMEOUT)
    await page.wait_for_timeout(3000) 
    captcha_element = page.locator(CAPTCHA_IMAGE_SELECTOR)
    logger.info(f"Waiting for CAPTCHA image ('{CAPTCHA_IMAGE_SELECTOR}') to be visible.")
    await captcha_element.wait_for(state="visible", timeout=ELEMENT_TIMEOUT)
    logger.info("Taking screenshot of CAPTCHA element and saving locally.")
    captcha_image_bytes = await captcha_element.screenshot()
    captcha_image_path.write_bytes(captcha_image_bytes)
    logger.info(f"CAPTCHA image saved to: {captcha_image_path.resolve()}")
    print(f"\n>>> CAPTCHA Image saved to: {captcha_image_path.resolve()}")
    print(">>> Please open the image, view the CAPTCHA, and type the characters below.")
    try:
        solution = await asyncio.to_thread(input, "Enter CAPTCHA solution: ")
        return solution.strip()
    except RuntimeError: logger.warning("Input prompt cancelled."); return None

async def _handle_final_popup_content(
    popup_page: Page, base_url_for_relative_links: str,
    output_dir_for_tender: Path, tender_file_id: str
) -> bool:
    logger.info(f"[{tender_file_id}] Processing final popup page: {popup_page.url}")
    await popup_page.wait_for_load_state("domcontentloaded", timeout=POPUP_PAGE_TIMEOUT)
    html_content_report = await popup_page.content()
    if html_content_report:
        processed_html_content = _remove_html_protection(html_content_report)
        safe_id_part = clean_filename_rot(tender_file_id)
        save_filename_html = f"ROT_{safe_id_part}_StageSummary.html"
        save_path_html = output_dir_for_tender / save_filename_html
        output_dir_for_tender.mkdir(parents=True, exist_ok=True) 
        save_path_html.write_text(processed_html_content, encoding='utf-8', errors='replace')
        logger.info(f"[{tender_file_id}] ✅ SAVED HTML FROM POPUP: {save_path_html.name}")
        return True
    logger.warning(f"[{tender_file_id}] No HTML content from final popup page."); return False

async def process_single_tender_detail_page(
    semaphore: asyncio.Semaphore, main_browser_context: BrowserContext,
    detail_page_url: str, site_base_url: str, 
    tender_display_id: str, list_page_num: int
):
    async with semaphore:
        log_prefix = f"[ListPage{list_page_num}_{clean_filename_rot(tender_display_id)}]"
        logger.info(f"{log_prefix} Starting detail processing for URL: {detail_page_url}")
        detail_page: Optional[Page] = None; popup_page: Optional[Page] = None
        tender_specific_output_dir = OUTPUT_DIR / clean_filename_rot(tender_display_id) # Changed from DOWNLOAD_DIR
        try:
            detail_page = await main_browser_context.new_page()
            await detail_page.goto(detail_page_url, wait_until="domcontentloaded", timeout=DETAIL_PAGE_TIMEOUT)
            logger.info(f"{log_prefix} Navigated to detail page: {detail_page.url}")
            summary_link_locator = detail_page.locator(SUMMARY_LINK_ON_DETAIL_PAGE).first
            await summary_link_locator.wait_for(state="visible", timeout=ELEMENT_TIMEOUT)
            logger.info(f"{log_prefix} Final summary link found. Clicking...")
            async with detail_page.expect_popup(timeout=POPUP_PAGE_TIMEOUT) as popup_info: await summary_link_locator.click()
            popup_page = await popup_info.value
            success = await _handle_final_popup_content(popup_page, site_base_url, tender_specific_output_dir, tender_display_id)
            if success: logger.info(f"{log_prefix} Successfully processed detail and popup.")
            else: logger.warning(f"{log_prefix} Failed to process content from popup.")
        except PlaywrightTimeout as pt_err:
            logger.error(f"{log_prefix} Timeout detail/popup for '{detail_page_url}': {pt_err}", exc_info=False)
            if detail_page and not detail_page.is_closed(): await detail_page.screenshot(path=OUTPUT_DIR / f"DEBUG_DETAIL_PG_TIMEOUT_{clean_filename_rot(tender_display_id)}_{SCRIPT_NAME_TAG}.png")
            if popup_page and not popup_page.is_closed(): await popup_page.screenshot(path=OUTPUT_DIR / f"DEBUG_POPUP_PG_TIMEOUT_{clean_filename_rot(tender_display_id)}_{SCRIPT_NAME_TAG}.png")
        except Exception as e: logger.error(f"{log_prefix} Error processing detail link '{detail_page_url}': {e}", exc_info=True)
        finally:
            if popup_page and not popup_page.is_closed(): await popup_page.close()
            if detail_page and not detail_page.is_closed(): await detail_page.close()

# Corrected function signature using 'Any' from typing
async def extract_table_data_from_current_list_page(page: Page, list_page_num: int, site_base_url_for_links: str) -> List[Dict[str, Any]]:
    extracted_data: list[dict[str, Any]] = []
    logger.info(f"[ListPage{list_page_num}] Extracting table data...")
    try:
        results_table_html = await page.locator(RESULTS_TABLE_SELECTOR).inner_html(timeout=ELEMENT_TIMEOUT)
        soup = BeautifulSoup(results_table_html, "html.parser"); rows = soup.select("tr[id^='informal']")
        if not rows:
            tbl_txt_el = soup.select_one(RESULTS_TABLE_SELECTOR);
            if tbl_txt_el and ("no records found" in tbl_txt_el.get_text(strip=True).lower() or "no tenders available" in tbl_txt_el.get_text(strip=True).lower()): 
                logger.info(f"[ListPage{list_page_num}] 'No records' text found in table."); return ["NO_RECORDS_FOUND_IN_TABLE"] # type: ignore
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
                extracted_data.append({
                    "list_page_num": str(list_page_num), "s_no_on_page": s_no, "tender_id": tender_id,
                    "title_and_ref_no": title_ref, "organisation_chain": org_chain, "tender_stage": tender_stage,
                    "view_status_link_href_relative": view_link_href_relative, "list_page_num_log": list_page_num
                })
            else: logger.warning(f"[ListPage{list_page_num}] Row {row_idx+1} has {len(cols)} cols, expected 6. Skipping.")
        logger.info(f"[ListPage{list_page_num}] Extracted {len(extracted_data)} items.")
    except PlaywrightTimeout: logger.error(f"[ListPage{list_page_num}] Timeout getting HTML of '{RESULTS_TABLE_SELECTOR}'."); await page.screenshot(path=OUTPUT_DIR / f"DEBUG_TABLE_EXTRACT_TIMEOUT_P{list_page_num}_{SCRIPT_NAME_TAG}.png")
    except Exception as e: logger.error(f"[ListPage{list_page_num}] Error extracting table data: {e}", exc_info=True)
    return extracted_data

async def run_headless_rot_scrape_orchestration(
    live_site_url: str, live_site_base_url_for_links: str, tender_status_to_select: str
    ):
    logger.info(f"--- Starting Headless ROT Scrape for {live_site_url} ---")
    all_extracted_tender_list_data: List[Dict[str, Any]] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True) 
        context = await browser.new_context(ignore_https_errors=True, accept_downloads=True)
        page = await context.new_page() 
        try:
            logger.info(f"Navigating to: {live_site_url}")
            await page.goto(live_site_url, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT)
            captcha_solution = await _save_captcha_and_get_solution(page, CAPTCHA_IMAGE_SAVE_PATH, tender_status_to_select)
            if not captcha_solution: logger.error("No CAPTCHA solution. Aborting."); return
            logger.info(f"CAPTCHA solution ('******') received. Submitting form...")
            await page.locator(CAPTCHA_TEXT_INPUT_SELECTOR).fill(captcha_solution)
            await page.locator(SEARCH_BUTTON_SELECTOR).click()
            logger.info(f"Waiting for initial results ('{RESULTS_TABLE_SELECTOR}') OR error.")
            await page.wait_for_selector(f"{RESULTS_TABLE_SELECTOR}, {ERROR_MESSAGE_SELECTORS}", state="visible", timeout=POST_SUBMIT_TIMEOUT)
            if not await page.locator(RESULTS_TABLE_SELECTOR).is_visible(timeout=2000):
                err_text = "Unknown (Results table not found)"; err_loc = page.locator(ERROR_MESSAGE_SELECTORS).first
                if await err_loc.is_visible(timeout=1000): err_text_content = await err_loc.text_content(); err_text = err_text_content.strip() if err_text_content else err_text
                logger.error(f"CAPTCHA submission failed: {err_text}"); await page.screenshot(path=OUTPUT_DIR / f"DEBUG_HEADLESS_CAPTCHA_SUBMIT_ERROR_{SCRIPT_NAME_TAG}.png")
                raise Exception(f"CAPTCHA/Search Error: {err_text}")
            logger.info("✅ Initial results table detected.")
            processed_list_pages = 0; detail_semaphore = asyncio.Semaphore(DETAIL_PROCESSING_CONCURRENCY); consecutive_empty_list_pages = 0
            while processed_list_pages < MAX_ROT_LIST_PAGES_TO_FETCH:
                processed_list_pages += 1; logger.info(f"--- Processing List Page: {processed_list_pages} ---")
                await page.wait_for_selector(RESULTS_TABLE_SELECTOR, state="visible", timeout=ELEMENT_TIMEOUT)
                current_page_items = await extract_table_data_from_current_list_page(page, processed_list_pages, live_site_base_url_for_links)
                if current_page_items == ["NO_RECORDS_FOUND_IN_TABLE"]: logger.info(f"'No records' on page {processed_list_pages}. Ending."); break # type: ignore
                if not current_page_items:
                    logger.info(f"No items on list page {processed_list_pages}."); consecutive_empty_list_pages += 1
                    if consecutive_empty_list_pages >= 2: logger.warning("Two consecutive empty list pages. Ending."); break
                else:
                    consecutive_empty_list_pages = 0; all_extracted_tender_list_data.extend(current_page_items)
                detail_tasks = []
                for item_data in current_page_items:
                    relative_href = item_data.get("view_status_link_href_relative")
                    if relative_href and relative_href != "N/A":
                        full_detail_url = urljoin(live_site_base_url_for_links, relative_href)
                        task = process_single_tender_detail_page(
                            detail_semaphore, context, full_detail_url, live_site_base_url_for_links, 
                            item_data.get("tender_id", f"Item{item_data.get('s_no_on_page','?')}_P{processed_list_pages}"), 
                            processed_list_pages )
                        detail_tasks.append(task)
                if detail_tasks: logger.info(f"Gathering {len(detail_tasks)} detail tasks for page {processed_list_pages}..."); await asyncio.gather(*detail_tasks); logger.info(f"Detail tasks for page {processed_list_pages} finished.")
                next_page_locator = page.locator(PAGINATION_LINK_SELECTOR)
                if await next_page_locator.is_visible(timeout=5000):
                    if processed_list_pages >= MAX_ROT_LIST_PAGES_TO_FETCH: logger.info(f"Max list pages ({MAX_ROT_LIST_PAGES_TO_FETCH}) reached."); break
                    logger.info(f"Clicking 'Next' for page {processed_list_pages + 1}."); await next_page_locator.click()
                    logger.info(f"Waiting {WAIT_AFTER_PAGINATION_CLICK_MS / 1000}s for next page..."); await page.wait_for_timeout(WAIT_AFTER_PAGINATION_CLICK_MS)
                else: logger.info(f"No 'Next' link after page {processed_list_pages}. End."); break
            logger.info(f"Finished list processing. Attempted {processed_list_pages} pages.")
            if all_extracted_tender_list_data:
                table_data_path = OUTPUT_DIR / EXTRACTED_TABLE_DATA_FILENAME;
                try:
                    with open(table_data_path, "w", encoding="utf-8") as f_json: json.dump(all_extracted_tender_list_data, f_json, indent=2, ensure_ascii=False)
                    logger.info(f"✅ Extracted list data for {len(all_extracted_tender_list_data)} items to: {table_data_path}")
                except Exception as e_json_save: logger.error(f"Failed to save list data to JSON: {e_json_save}")
            else: logger.info("No table data extracted to save.")
        except Exception as e: logger.critical(f"Major error in headless ROT scrape: {e}", exc_info=True)
        finally:
            if context: await context.close()
            if browser: await browser.close()
            logger.info("Playwright resources for headless ROT closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless ROT Scraper with console CAPTCHA input.")
    parser.add_argument("--site_url", default=LIVE_SITE_ROT_SEARCH_URL, help="Full URL of the ROT search page.") # Use LIVE_SITE_ROT_SEARCH_URL
    parser.add_argument("--base_url", default=LIVE_SITE_BASE_URL, help="Base URL for resolving relative links.") # Use LIVE_SITE_BASE_URL
    parser.add_argument("--tender_status", default=DEFAULT_TENDER_STATUS, help=f"Tender status value (default: {DEFAULT_TENDER_STATUS}).") # This one was correct
    args = parser.parse_args()
 
    print("--- Headless ROT Scraper with Console CAPTCHA Input ---")
    # ... (print other args/config) ...
    print(f"Output (HTML summaries & debug) will be in: {OUTPUT_DIR.resolve()}")
    print(f"Extracted list data JSON: {OUTPUT_DIR / EXTRACTED_TABLE_DATA_FILENAME}")
    print("-" * 70)
    
    asyncio.run(run_headless_rot_scrape_orchestration(
        args.site_url,
        args.base_url,
        args.tender_status
    ))
