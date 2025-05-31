import asyncio
import base64
from pathlib import Path
import re
import logging
import json # Added for JSON output
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup, Tag

# --- Configuration ---
LIVE_SITE_ROT_SEARCH_URL = "https://eprocure.gov.in/eprocure/app?page=WebTenderStatusLists&service=page"
LIVE_SITE_BASE_URL = "https://eprocure.gov.in/eprocure/app/"

TENDER_STATUS_DROPDOWN_SELECTOR_LIVE = "#tenderStatus"
CAPTCHA_IMAGE_SELECTOR_LIVE = "#captchaImage"
CAPTCHA_TEXT_INPUT_SELECTOR_LIVE = "#captchaText"
SEARCH_BUTTON_SELECTOR_LIVE = "#Search"
RESULTS_TABLE_SELECTOR_LIVE = "#tabList"
VIEW_STATUS_LINK_IN_RESULTS_LIVE = "table#tabList tr[id^='informal'] a[id^='view']"
SUMMARY_LINK_SELECTOR_LIVE = "a#DirectLink_0:has-text('Click link to view the all stage summary Details')"
ERROR_MESSAGE_SELECTORS_LIVE = ".error_message, .errormsg, #msgDiv"
PAGINATION_LINK_SELECTOR_LIVE = "a#loadNext"

MAX_ROT_LIST_PAGES_TO_FETCH = 3
DETAIL_PROCESSING_CONCURRENCY = 3
WAIT_AFTER_PAGINATION_CLICK_MS = 7000

DOWNLOAD_DIR = Path(__file__).parent / "standalone_rot_output_paged_concurrent"
CAPTCHA_DEBUG_FILENAME = "live_captcha_debug.png"
SCRIPT_NAME_TAG = Path(__file__).stem
EXTRACTED_TABLE_DATA_FILENAME = "extracted_tender_list_data.json" # New

PAGE_LOAD_TIMEOUT = 60000
DETAIL_PAGE_TIMEOUT = 75000
ELEMENT_TIMEOUT = 20000
POST_SUBMIT_TIMEOUT = 30000

logger = logging.getLogger("standalone_live_rot_logger_paged_concurrent")
# (Logging setup same as before)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    if logger.hasHandlers(): logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s'))
    logger.addHandler(ch)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _remove_html_protection(html_content: str) -> str:
    # (Same as before)
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
        body_tag = match.group(0)
        cleaned_body_tag = re.sub(r'\s*onpageshow\s*=\s*".*?"', '', body_tag, flags=re.IGNORECASE)
        cleaned_body_tag = re.sub(r"\s*onpageshow\s*=\s*'.*?'", '', cleaned_body_tag, flags=re.IGNORECASE)
        return cleaned_body_tag
    modified_content = re.sub(r'<body[^>]*>', remove_onpageshow, modified_content, count=1, flags=re.IGNORECASE)
    if modified_content != html_content: logger.info("Applied modifications to downloaded HTML via _remove_html_protection.")
    else: logger.debug("No protection patterns matched by _remove_html_protection or content unchanged.")
    return modified_content

def clean_filename_rot(filename: str) -> str:
    # (Same as before)
    cleaned = re.sub(r'[\\/*?:"<>|\r\n\t]+', '_', filename)
    cleaned = re.sub(r'[\s_]+', '_', cleaned).strip('_')
    return cleaned[:150] if len(cleaned) > 150 else cleaned

async def _click_final_summary_link_and_process_popup(
    page_with_final_link: Page, base_url_for_relative_links: str,
    output_dir: Path, tender_unique_id: str
) -> bool:
    # (Same as before)
    logger.info(f"[{tender_unique_id}] On 'Tender Status Details' page. Finding FINAL summary link ('{SUMMARY_LINK_SELECTOR_LIVE}').")
    final_summary_link_locator = page_with_final_link.locator(SUMMARY_LINK_SELECTOR_LIVE).first
    try:
        await final_summary_link_locator.wait_for(state="visible", timeout=ELEMENT_TIMEOUT)
        logger.info(f"[{tender_unique_id}] FINAL summary link is visible. Clicking it.")
        temp_page_for_report: Page | None = None
        try:
            async with page_with_final_link.expect_popup(timeout=DETAIL_PAGE_TIMEOUT) as popup_info:
                await final_summary_link_locator.click()
            temp_page_for_report = await popup_info.value
            await temp_page_for_report.wait_for_load_state("domcontentloaded", timeout=DETAIL_PAGE_TIMEOUT)
            html_content_report = await temp_page_for_report.content()
            if html_content_report:
                processed_html_content = _remove_html_protection(html_content_report)
                safe_id_part = clean_filename_rot(tender_unique_id)
                save_filename_html = f"ROT_{safe_id_part}_StageSummary.html" # Dynamic filename
                save_path_html = output_dir / save_filename_html
                save_path_html.write_text(processed_html_content, encoding='utf-8', errors='replace')
                logger.info(f"[{tender_unique_id}] ✅ PROCESSED HTML FROM POPUP SAVED: {save_path_html.name}")
                return True
            return False # No content
        except PlaywrightTimeout as pt_err:
            logger.error(f"[{tender_unique_id}] ❌ Timeout with popup: {pt_err}", exc_info=False)
            if temp_page_for_report and not temp_page_for_report.is_closed():
                 await temp_page_for_report.screenshot(path=output_dir / f"DEBUG_POPUP_TIMEOUT_{clean_filename_rot(tender_unique_id)}_{SCRIPT_NAME_TAG}.png")
            return False
        finally:
            if temp_page_for_report and not temp_page_for_report.is_closed(): await temp_page_for_report.close()
    except PlaywrightTimeout:
        logger.error(f"[{tender_unique_id}] Timeout waiting for FINAL summary link '{SUMMARY_LINK_SELECTOR_LIVE}' to be visible.")
        await page_with_final_link.screenshot(path=output_dir / f"DEBUG_FINAL_LINK_TIMEOUT_{clean_filename_rot(tender_unique_id)}_{SCRIPT_NAME_TAG}.png")
        return False
    except Exception as e:
        logger.error(f"[{tender_unique_id}] ❌ Unexpected error in FINAL link/popup stage: {e}", exc_info=True)
        return False

async def process_single_detail_link(
    semaphore: asyncio.Semaphore, original_context: BrowserContext,
    view_status_link_href: str, base_url: str, output_dir: Path,
    tender_display_id: str, list_page_num: int
):
    # (Same as before)
    async with semaphore:
        log_prefix = f"[ListPage{list_page_num}_{clean_filename_rot(tender_display_id)}]"
        logger.info(f"{log_prefix} Starting detail processing for: {view_status_link_href}")
        detail_page: Page | None = None
        try:
            detail_page = await original_context.new_page()
            await detail_page.goto(view_status_link_href, wait_until="domcontentloaded", timeout=DETAIL_PAGE_TIMEOUT)
            logger.info(f"{log_prefix} Navigated to 'Tender Status Details' page: {detail_page.url}")
            success = await _click_final_summary_link_and_process_popup(
                detail_page, base_url, output_dir,
                f"L{list_page_num}_{clean_filename_rot(tender_display_id)}"
            )
            if success: logger.info(f"{log_prefix} Successfully processed detail.")
            else: logger.warning(f"{log_prefix} Failed to process detail.")
        except PlaywrightTimeout as pt_err:
            logger.error(f"{log_prefix} Timeout in detail processing for '{view_status_link_href}': {pt_err}", exc_info=False)
            if detail_page and not detail_page.is_closed():
                await detail_page.screenshot(path=output_dir / f"DEBUG_DETAIL_TIMEOUT_{clean_filename_rot(tender_display_id)}_{SCRIPT_NAME_TAG}.png")
        except Exception as e:
            logger.error(f"{log_prefix} Error processing detail link '{view_status_link_href}': {e}", exc_info=True)
        finally:
            if detail_page and not detail_page.is_closed(): await detail_page.close()

# --- NEW FUNCTION ---
async def extract_table_data_from_list_page(page: Page, list_page_num: int) -> list[dict[str, str]]:
    extracted_data: list[dict[str, str]] = []
    logger.info(f"[ListPage{list_page_num}] Extracting table data from results list...")
    try:
        results_table_html = await page.locator(RESULTS_TABLE_SELECTOR_LIVE).inner_html(timeout=ELEMENT_TIMEOUT)
        soup = BeautifulSoup(results_table_html, "html.parser")
        
        # Find all tender rows (tr elements with id starting with 'informal')
        # The first row inside #tabList might be headers, so skip if not id^=informal
        # The HTML provided shows rows directly inside #tabList like <tr class="even" id="informal">
        # and then <tr class="odd" id="informal_0">. So id^=informal should cover both.
        rows = soup.select("tr[id^='informal']")

        if not rows:
            logger.warning(f"[ListPage{list_page_num}] No data rows (tr[id^='informal']) found in #tabList.")
            return []

        for row_idx, row in enumerate(rows):
            cols = row.find_all("td")
            if len(cols) == 6: # Expecting 6 columns based on your example
                s_no = cols[0].get_text(strip=True)
                tender_id = cols[1].get_text(strip=True)
                title_ref = cols[2].get_text(strip=True)
                org_chain = cols[3].get_text(strip=True)
                tender_stage = cols[4].get_text(strip=True)
                # For status, just note if view link exists
                status_link_exists = "View Link Present" if cols[5].find("a", id=re.compile(r"^view_?\d*$")) else "No View Link"
                
                extracted_data.append({
                    "list_page_num": str(list_page_num),
                    "s_no_on_page": s_no,
                    "tender_id": tender_id,
                    "title_and_ref_no": title_ref,
                    "organisation_chain": org_chain,
                    "tender_stage": tender_stage,
                    "status_column_info": status_link_exists
                })
            else:
                logger.warning(f"[ListPage{list_page_num}] Row {row_idx} has {len(cols)} columns, expected 6. Skipping. Row HTML: {row.prettify()[:200]}...")
        logger.info(f"[ListPage{list_page_num}] Extracted {len(extracted_data)} items from table.")
    except PlaywrightTimeout:
        logger.error(f"[ListPage{list_page_num}] Timeout getting innerHTML of results table '{RESULTS_TABLE_SELECTOR_LIVE}'.")
        await page.screenshot(path=DOWNLOAD_DIR / f"DEBUG_TABLE_EXTRACT_TIMEOUT_P{list_page_num}_{SCRIPT_NAME_TAG}.png")
    except Exception as e:
        logger.error(f"[ListPage{list_page_num}] Error extracting table data: {e}", exc_info=True)
    return extracted_data


class LiveSiteInteractor:
    def __init__(self, browser_obj: BrowserContext.browser):
        # (Same as before)
        self.browser_instance = browser_obj
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    async def _ensure_context_and_page(self):
        # (Same as before)
        if self.context is None or self.page is None or self.page.is_closed():
            if self.context and not self.context.is_closed(): await self.context.close() # type: ignore
            logger.info("Creating new Playwright context and page for live site.")
            self.context = await self.browser_instance.new_context(ignore_https_errors=True, accept_downloads=True)
            self.page = await self.context.new_page()
        else: logger.info("Reusing existing Playwright context and page.")

    async def fetch_captcha_from_live_site(self, site_url: str, tender_status_value: str):
        # (Same as before)
        await self._ensure_context_and_page()
        assert self.page is not None
        try:
            logger.info(f"Navigating to live site for CAPTCHA: {site_url}")
            await self.page.goto(site_url, wait_until="domcontentloaded", timeout=PAGE_LOAD_TIMEOUT)
            logger.info(f"Selecting tender status '{tender_status_value}'.")
            await self.page.locator(TENDER_STATUS_DROPDOWN_SELECTOR_LIVE).select_option(value=tender_status_value)
            await self.page.wait_for_timeout(3000)
            captcha_element = self.page.locator(CAPTCHA_IMAGE_SELECTOR_LIVE)
            await captcha_element.wait_for(state="visible", timeout=ELEMENT_TIMEOUT)
            captcha_image_bytes = await captcha_element.screenshot()
            (DOWNLOAD_DIR / CAPTCHA_DEBUG_FILENAME).write_bytes(captcha_image_bytes)
            logger.info(f"Debug CAPTCHA saved: {DOWNLOAD_DIR / CAPTCHA_DEBUG_FILENAME}")
            return {"success": True, "captcha_image_uri": f"data:image/png;base64,{base64.b64encode(captcha_image_bytes).decode('utf-8')}"}
        except Exception as e:
            error_msg = f"Error/Timeout fetching CAPTCHA: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if self.page and not self.page.is_closed(): await self.page.screenshot(path=DOWNLOAD_DIR / f"DEBUG_CAPTCHA_FETCH_FAIL_{SCRIPT_NAME_TAG}.png")
            return {"success": False, "error": error_msg}

    async def submit_captcha_and_scrape_all_results(self, tender_status_value: str, captcha_solution: str):
        if self.page is None or self.page.is_closed() or self.context is None:
            return {"success": False, "error": "Live site page/context not available. Fetch CAPTCHA first."}
        
        main_page = self.page
        all_extracted_tenders_data: list[dict[str,str]] = [] # Initialize list to store all table data
        
        try:
            logger.info("Filling CAPTCHA and submitting form.")
            await main_page.locator(TENDER_STATUS_DROPDOWN_SELECTOR_LIVE).select_option(value=tender_status_value)
            await main_page.locator(CAPTCHA_TEXT_INPUT_SELECTOR_LIVE).fill(captcha_solution)
            await main_page.locator(SEARCH_BUTTON_SELECTOR_LIVE).click()
            
            logger.info(f"Waiting for initial results ('{RESULTS_TABLE_SELECTOR_LIVE}') OR error.")
            combined_selectors = f"{RESULTS_TABLE_SELECTOR_LIVE}, {ERROR_MESSAGE_SELECTORS_LIVE}"
            try:
                await main_page.wait_for_selector(combined_selectors, state="visible", timeout=POST_SUBMIT_TIMEOUT)
            except PlaywrightTimeout:
                logger.error("Timeout: Neither results table nor error message appeared after CAPTCHA submission.")
                await main_page.screenshot(path=DOWNLOAD_DIR / f"DEBUG_INITIAL_LOAD_TIMEOUT_{SCRIPT_NAME_TAG}.png")
                return {"success": False, "error": "Timeout waiting for initial results/error."}

            if not await main_page.locator(RESULTS_TABLE_SELECTOR_LIVE).is_visible(timeout=1000):
                # (Error handling same as before)
                error_text = "Unknown error (Results table not found, error message also not found or not recognized)."
                if await main_page.locator(ERROR_MESSAGE_SELECTORS_LIVE).count() > 0:
                     first_error_element = main_page.locator(ERROR_MESSAGE_SELECTORS_LIVE).first
                     if await first_error_element.is_visible(timeout=500):
                        error_text_content = await first_error_element.text_content()
                        if error_text_content: error_text = error_text_content.strip()
                logger.error(f"CAPTCHA submission failed or error on page: {error_text}")
                await main_page.screenshot(path=DOWNLOAD_DIR / f"DEBUG_CAPTCHA_SUBMIT_ERROR_{SCRIPT_NAME_TAG}.png")
                return {"success": False, "error": f"CAPTCHA/Search Error: {error_text}"}
            
            logger.info("✅ Initial results table detected. CAPTCHA successful.")
            
            processed_list_pages = 0
            semaphore = asyncio.Semaphore(DETAIL_PROCESSING_CONCURRENCY)

            while processed_list_pages < MAX_ROT_LIST_PAGES_TO_FETCH:
                processed_list_pages += 1
                logger.info(f"--- Processing List Page: {processed_list_pages} ---")
                
                await main_page.wait_for_selector(RESULTS_TABLE_SELECTOR_LIVE, state="visible", timeout=ELEMENT_TIMEOUT)

                # --- Extract table data from current list page ---
                current_page_table_data = await extract_table_data_from_list_page(main_page, processed_list_pages)
                all_extracted_tenders_data.extend(current_page_table_data)
                # --- End: Extract table data ---

                view_status_links_locators = await main_page.locator(VIEW_STATUS_LINK_IN_RESULTS_LIVE).all()
                if not view_status_links_locators:
                    logger.info(f"No 'View Status' links found on list page {processed_list_pages}. Ending.")
                    break
                
                logger.info(f"Found {len(view_status_links_locators)} 'View Status' links for detail processing on list page {processed_list_pages}.")
                
                tasks = []
                # Using the extracted table data to get a better display ID
                for i, link_locator in enumerate(view_status_links_locators):
                    href = await link_locator.get_attribute("href")
                    if not href:
                        logger.warning(f"Link {i+1} on page {processed_list_pages} has no href. Skipping.")
                        continue
                    
                    full_href = urljoin(LIVE_SITE_BASE_URL, href)
                    
                    tender_display_id_for_log = f"Item{i+1}" # Default
                    # Try to find corresponding tender_id from extracted table data for better logging
                    # This assumes view_status_links_locators and current_page_table_data (if parsed before links) are in same order
                    if i < len(current_page_table_data) and "tender_id" in current_page_table_data[i]:
                        tender_display_id_for_log = current_page_table_data[i]["tender_id"]
                    else: # Fallback if table data extraction failed or was out of sync
                        try:
                            parent_row = link_locator.locator("xpath=ancestor::tr[contains(@id, 'informal')]")
                            if await parent_row.count() > 0:
                                tender_id_cell = parent_row.locator("td").nth(1)
                                if await tender_id_cell.count() > 0:
                                    tid_text = await tender_id_cell.text_content()
                                    if tid_text: tender_display_id_for_log = tid_text.strip()
                        except Exception: pass


                    task = asyncio.create_task(process_single_detail_link(
                        semaphore, self.context, full_href, LIVE_SITE_BASE_URL,
                        DOWNLOAD_DIR, tender_display_id_for_log, processed_list_pages
                    ))
                    tasks.append(task)
                
                if tasks:
                    logger.info(f"Waiting for {len(tasks)} concurrent detail tasks for list page {processed_list_pages}...")
                    await asyncio.gather(*tasks)
                    logger.info(f"All detail tasks for list page {processed_list_pages} finished.")
                
                next_page_locator = main_page.locator(PAGINATION_LINK_SELECTOR_LIVE)
                if await next_page_locator.is_visible(timeout=5000):
                    if processed_list_pages >= MAX_ROT_LIST_PAGES_TO_FETCH:
                        logger.info(f"Reached max list pages ({MAX_ROT_LIST_PAGES_TO_FETCH}). Stopping.")
                        break
                    logger.info(f"Clicking 'Next' for list page {processed_list_pages + 1}.")
                    await next_page_locator.click()
                    logger.info(f"Waiting {WAIT_AFTER_PAGINATION_CLICK_MS / 1000}s for next page results...")
                    await main_page.wait_for_timeout(WAIT_AFTER_PAGINATION_CLICK_MS) # Wait for AJAX
                    # Add a check to see if #tabList content actually changed
                    # e.g., by comparing the first tender ID before and after the click
                else:
                    logger.info("No 'Next' page link. End of results.")
                    break
            
            # --- Save extracted table data to JSON ---
            if all_extracted_tenders_data:
                table_data_path = DOWNLOAD_DIR / EXTRACTED_TABLE_DATA_FILENAME
                try:
                    with open(table_data_path, "w", encoding="utf-8") as f:
                        json.dump(all_extracted_tenders_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"✅ Extracted table data for {len(all_extracted_tenders_data)} items saved to: {table_data_path}")
                except Exception as e_json:
                    logger.error(f"Failed to save extracted table data to JSON: {e_json}")
            else:
                logger.info("No table data was extracted from any list page.")
            # --- End: Save JSON ---

            logger.info(f"Finished. Processed {processed_list_pages} list pages.")
            return {"success": True, "message": f"Scraping finished."}

        except Exception as e:
            # (Error handling same as before)
            error_msg = f"Major error during scraping: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if main_page and not main_page.is_closed(): await main_page.screenshot(path=DOWNLOAD_DIR / f"DEBUG_MAIN_SCRAPE_ERROR_{SCRIPT_NAME_TAG}.png")
            return {"success": False, "error": error_msg}

    async def close(self):
        # (Same as before)
        if self.page and not self.page.is_closed(): await self.page.close()
        if self.context and not self.context.is_closed(): await self.context.close() # type: ignore
        logger.info("Live site interactor resources closed.")


async def run_standalone_test():
    # (Same startup checks as before, ensure DOWNLOAD_DIR is also created before logger uses it if file handler added)
    required_selectors = [RESULTS_TABLE_SELECTOR_LIVE, VIEW_STATUS_LINK_IN_RESULTS_LIVE, SUMMARY_LINK_SELECTOR_LIVE, PAGINATION_LINK_SELECTOR_LIVE]
    if LIVE_SITE_ROT_SEARCH_URL == "YOUR_GOVERNMENT_SITE_ROT_SEARCH_PAGE_URL" or \
       any(not selector or "YOUR_" in selector.upper() for selector in required_selectors):
        print("CRITICAL ERROR: Update placeholder URLs and ALL CRITICAL SELECTORS in the Python script.")
        return

    async with async_playwright() as p:
        # (Rest of run_standalone_test same as before)
        browser_obj = await p.chromium.launch(headless=True)
        live_site_handler = LiveSiteInteractor(browser_obj)
        ui_page = await browser_obj.new_page()

        async def py_fetch_live_captcha(site_key_from_ui: str, tender_status_from_ui_value: str):
            return await live_site_handler.fetch_captcha_from_live_site(LIVE_SITE_ROT_SEARCH_URL, tender_status_from_ui_value)

        async def py_submit_captcha_and_scrape(site_key_from_ui: str, tender_status_from_ui_value: str, captcha_solution: str):
            return await live_site_handler.submit_captcha_and_scrape_all_results(tender_status_from_ui_value, captcha_solution)

        await ui_page.expose_function("pyFetchLiveCaptcha", py_fetch_live_captcha)
        await ui_page.expose_function("pySubmitCaptchaAndScrape", py_submit_captcha_and_scrape)

        local_form_path = Path(__file__).parent / "standalone_form.html"
        await ui_page.goto(f"file://{local_form_path.resolve()}")
        logger.info(f"Local UI loaded. Max list pages: {MAX_ROT_LIST_PAGES_TO_FETCH}, Detail Concurrency: {DETAIL_PROCESSING_CONCURRENCY}")

        try:
            while not ui_page.is_closed() and browser_obj.is_connected():
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError): logger.info("Test interrupted.")
        finally:
            logger.info("Closing resources...")
            await live_site_handler.close()
            if browser_obj.is_connected(): await browser_obj.close()
            logger.info("Standalone test finished.")


if __name__ == "__main__":
    # (Same as before)
    print("--- Standalone Paged Concurrent ROT Scraper with Table Data Extraction ---")
    print("IMPORTANT: Verify ALL configuration constants and CSS selectors at the top of the script.")
    print(f"Max List Pages to Fetch: {MAX_ROT_LIST_PAGES_TO_FETCH}")
    print(f"Detail Processing Concurrency: {DETAIL_PROCESSING_CONCURRENCY}")
    print(f"Wait after pagination click: {WAIT_AFTER_PAGINATION_CLICK_MS / 1000}s")
    print(f"Output will be in: {DOWNLOAD_DIR.resolve()}")
    print(f"Extracted table data will be in: {DOWNLOAD_DIR / EXTRACTED_TABLE_DATA_FILENAME}")
    print("-" * 70)
    asyncio.run(run_standalone_test())
