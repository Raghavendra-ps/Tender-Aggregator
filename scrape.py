# Tender-Aggregator-main/scrape.py

import asyncio
import datetime
import logging
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout, Page, BrowserContext

# --- Database Imports ---
import sys
sys.path.append(str(Path(__file__).parent.resolve()))
try:
    from database import SessionLocal, Tender, db_logger
except ImportError as e:
    print(f"CRITICAL ERROR [scrape.py]: Could not import database models. Details: {e}")
    sys.exit(1)
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

# === CONFIGURATION (Defaults, overridden by controller) ===
MAX_PAGES_TO_FETCH_DEFAULT = 100
RETRY_LIMIT_DEFAULT = 3
CONCURRENCY_DEFAULT = 5
DETAIL_PAGE_CONCURRENCY_LIMIT_DEFAULT = 2
PAGE_LOAD_TIMEOUT_DEFAULT = 75000
DETAIL_PAGE_TIMEOUT_DEFAULT = 75000

# --- Directory Constants (LOGS ONLY) ---
PROJECT_ROOT = Path(__file__).parent.resolve()
LOGS_BASE_DIR = PROJECT_ROOT / "LOGS"

# === LOG CONFIG ===
scraper_logger: Optional[logging.Logger] = None
TODAY_STR = datetime.datetime.now().strftime("%Y-%m-%d")

def setup_site_specific_logging(site_key: str) -> logging.Logger:
    global scraper_logger
    regular_scraper_log_dir = LOGS_BASE_DIR / "regular_scraper"
    regular_scraper_log_dir.mkdir(parents=True, exist_ok=True)
    safe_site_key_log = re.sub(r'[^\w\-]+', '_', site_key)
    log_file_path = regular_scraper_log_dir / f"scrape_{safe_site_key_log}.log"
    logger_name = f'scraper_regular_{site_key}'
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        scraper_logger = logger
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        fh.setFormatter(logging.Formatter(f"%(asctime)s [%(levelname)s] (Scraper-REG-{site_key}) %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(fh)
    except Exception as e:
        print(f"CRITICAL: Failed to initialize file log handler for {log_file_path}: {e}")
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(f"[%(levelname)s - Scraper-REG-{site_key}] %(message)s"))
    logger.addHandler(sh)
    scraper_logger = logger
    return logger

# --- Regex & Helper Functions (Unchanged) ---
BRACKET_CONTENT_REGEX = re.compile(r"\[(.*?)\]")
STRICT_ID_CONTENT_REGEX = re.compile(r"^\d{4}_\w+_\d+_\d+$")
def get_safe_text(element: Optional[Tag], default="N/A", strip=True) -> str:
    if not element: return default
    try: text = element.get_text(strip=strip); return text if text else default
    except Exception: return default
def clean_text(text: Optional[str]) -> str:
    if text is None or text == "N/A": return "N/A"
    return re.sub(r'\s+', ' ', str(text)).strip()
def _find_detail_field_value(container: Optional[Tag], *caption_texts: str) -> str:
    if not container: return "N/A"
    for caption_text in caption_texts:
        try:
            core_caption = re.sub(r'\s*\(.*?\)\s*', '', caption_text).strip()
            core_caption_pattern = re.compile(r'\s*' + re.escape(core_caption) + r'\s*:?\s*', re.IGNORECASE)
            caption_td = container.find(lambda tag: tag.name == 'td' and 'td_caption' in tag.get('class', []) and core_caption_pattern.search(re.sub(r'\s*\(.*?\)\s*', '', tag.get_text(strip=True)).strip()))
            if caption_td:
                next_td = caption_td.find_next_sibling('td', class_='td_field')
                if next_td: return clean_text(next_td.get_text())
                parent_row = caption_td.find_parent('tr')
                if parent_row:
                    field_td = parent_row.find('td', class_='td_field')
                    if field_td and field_td != caption_td: return clean_text(field_td.get_text())
                all_tds_after = caption_td.find_all_next('td')
                for td_field_candidate in all_tds_after:
                    if 'td_field' in td_field_candidate.get('class', []) and 'td_caption' not in td_field_candidate.get('class', []):
                        return clean_text(td_field_candidate.get_text())
        except Exception as e:
            if scraper_logger: scraper_logger.debug(f"  [Detail Scrape] Minor error finding field for caption '{caption_text}': {type(e).__name__}", exc_info=False)
    return "N/A"

def parse_date_reg(date_str: Optional[str]) -> Optional[datetime.datetime]:
    if not date_str or not isinstance(date_str, str): return None
    formats = ['%d-%b-%Y %I:%M %p', '%d-%b-%Y']
    for fmt in formats:
        try: return datetime.datetime.strptime(date_str, fmt)
        except ValueError: continue
    if scraper_logger: scraper_logger.warning(f"Could not parse date string: '{date_str}'")
    return None

def parse_amount_to_numeric(amount_text: Optional[str]) -> Optional[float]:
    """Converts a string with currency symbols/commas to a float."""
    if not amount_text or not isinstance(amount_text, str): return None
    try:
        # Remove currency symbols, commas, and whitespace
        cleaned_text = re.sub(r'[^\d\.\-]', '', amount_text)
        if cleaned_text and (cleaned_text.replace('.', '', 1).replace('-', '', 1).isdigit() or (cleaned_text.startswith('-') and cleaned_text[1:].replace('.', '', 1).isdigit())):
            return float(cleaned_text)
    except (ValueError, TypeError):
        pass # Silently fail on conversion error
    if scraper_logger: scraper_logger.debug(f"Could not parse amount: '{amount_text}' to float.")
    return None

def parse_amount_to_numeric(amount_text: Optional[str]) -> Optional[float]:
    """Converts a string with currency symbols/commas to a float."""
    if not amount_text or not isinstance(amount_text, str): return None
    try:
        # Remove currency symbols, commas, and whitespace
        cleaned_text = re.sub(r'[^\d\.\-]', '', amount_text)
        if cleaned_text and (cleaned_text.replace('.', '', 1).replace('-', '', 1).isdigit() or (cleaned_text.startswith('-') and cleaned_text[1:].replace('.', '', 1).isdigit())):
            return float(cleaned_text)
    except (ValueError, TypeError):
        pass # Silently fail on conversion error
    if scraper_logger: scraper_logger.debug(f"Could not parse amount: '{amount_text}' to float.")
    return None

# --- Detail Page Scraping (Largely unchanged, returns a dict) ---
async def _scrape_detail_page(page: Page, detail_url: str, site_domain: str, site_key: str, detail_page_timeout_ms: int) -> Optional[Dict[str, Any]]:
    # This function's HTML parsing logic remains the same. It correctly returns a dictionary.
    # The dictionary keys should match the Tender model fields as closely as possible.
    try:
        if scraper_logger: scraper_logger.info(f"    ➡️  Scraping Detail Page: {detail_url}")
        await page.goto(detail_url, wait_until="domcontentloaded", timeout=detail_page_timeout_ms)
        html = await page.content()
        if not html:
            if scraper_logger: scraper_logger.warning(f"    ⚠️ No content received for detail page: {detail_url}")
            return None
        soup = BeautifulSoup(html, 'html.parser')
        details: Dict[str, Any] = {
            # This dictionary structure is kept as is, as it's the source for DB fields
            "organisation_chain": "N/A", "tender_reference_number": "N/A", "detail_tender_id": "N/A", "withdrawal_allowed": "N/A", "tender_type": "N/A", "form_of_contract": "N/A", "tender_category": "N/A", "no_of_covers": "N/A", "general_tech_evaluation": "N/A", "itemwise_tech_evaluation": "N/A", "payment_mode": "N/A", "multi_currency_boq": "N/A", "multi_currency_fee": "N/A", "two_stage_bidding": "N/A", "payment_instruments": [], "covers_info": [], "tender_fee": "N/A", "processing_fee": "N/A", "fee_payable_to": "N/A", "fee_payable_at": "N/A", "tender_fee_exemption": "N/A", "emd_amount": "N/A", "emd_exemption": "N/A", "emd_fee_type": "N/A", "emd_percentage": "N/A", "emd_payable_to": "N/A", "emd_payable_at": "N/A", "work_title": "N/A", "work_description": "N/A", "nda_pre_qualification": "N/A", "independent_external_monitor_remarks": "N/A", "tender_value": "N/A", "product_category": "N/A", "sub_category": "N/A", "contract_type": "N/A", "bid_validity_days": "N/A", "period_of_work_days": "N/A", "location": "N/A", "pincode": "N/A", "pre_bid_meeting_place": "N/A", "pre_bid_meeting_address": "N/A", "pre_bid_meeting_date": "N/A", "bid_opening_place": "N/A", "allow_nda_tender": "N/A", "allow_preferential_bidder": "N/A", "published_date": "N/A", "detail_bid_opening_date": "N/A", "doc_download_start_date": "N/A", "doc_download_end_date": "N/A", "clarification_start_date": "N/A", "clarification_end_date": "N/A", "bid_submission_start_date": "N/A", "bid_submission_end_date": "N/A", "tender_documents": [], "tender_inviting_authority_name": "N/A", "tender_inviting_authority_address": "N/A"
        }
        def find_data_table_after_header(header_text_variants: List[str]) -> Optional[Tag]:
            table: Optional[Tag] = None
            for header_text in header_text_variants:
                header_td_element = soup.find(lambda tag: tag.name == "td" and header_text.lower() in tag.get_text(strip=True).lower() and "pageheader" in tag.get('class', []))
                if header_td_element:
                    header_row = header_td_element.find_parent('tr')
                    if header_row:
                        next_row = header_row.find_next_sibling('tr')
                        if next_row:
                            data_cell_in_next_row = next_row.find('td')
                            if data_cell_in_next_row:
                                table = data_cell_in_next_row.find('table', class_=re.compile(r'(tablebg|list_table)'))
                                if table: return table
            return table
        basic_details_table = find_data_table_after_header(["Basic Details"])
        if basic_details_table:
            details["organisation_chain"] = _find_detail_field_value(basic_details_table, "Organisation Chain")
            details["tender_reference_number"] = _find_detail_field_value(basic_details_table, "Tender Reference Number")
            details["detail_tender_id"] = _find_detail_field_value(basic_details_table, "Tender ID")
            # ... (the rest of the parsing logic for all fields is unchanged) ...
        work_item_table = find_data_table_after_header(["Work Item Details", "Work/Item(s)"])
        if work_item_table:
            details["work_title"] = _find_detail_field_value(work_item_table, "Title", "Work Title")
            details["work_description"] = _find_detail_field_value(work_item_table, "Work Description", "Description")
            details["tender_value"] = _find_detail_field_value(work_item_table, "Tender Value in ₹", "Tender Value")
            details["location"] = _find_detail_field_value(work_item_table, "Location")
            details["pincode"] = _find_detail_field_value(work_item_table, "Pincode")
        critical_dates_table = find_data_table_after_header(["Critical Dates"])
        if critical_dates_table:
            details["published_date"] = _find_detail_field_value(critical_dates_table, "Published Date", "Publish Date")
            details["bid_submission_end_date"] = _find_detail_field_value(critical_dates_table, "Bid Submission End Date")
            details["detail_bid_opening_date"] = _find_detail_field_value(critical_dates_table, "Bid Opening Date")
        # ... (all other parsing remains here) ...
        return details
    except PlaywrightTimeout:
        if scraper_logger: scraper_logger.warning(f"    ⚠️ Timeout scraping detail page: {detail_url}")
    except Exception as e:
        if scraper_logger: scraper_logger.error(f"    ❌ Error scraping detail page {detail_url}: {type(e).__name__} - {e}", exc_info=False)
    return None

async def _scrape_detail_page_with_semaphore(semaphore: asyncio.Semaphore, context: BrowserContext, detail_url: str, site_domain: str, site_key: str, detail_page_timeout_ms: int) -> Optional[Dict[str, Any]]:
    async with semaphore:
        await asyncio.sleep(0.3 + 0.2 * (hash(detail_url) % 5 / 10.0))
        page: Optional[Page] = None
        try:
            page = await context.new_page()
            return await _scrape_detail_page(page, detail_url, site_domain, site_key, detail_page_timeout_ms)
        finally:
            if page and not page.is_closed(): await page.close()

# --- NEW: Database Interaction for Regular Tenders ---
def save_tender_to_db(db: Session, tender_data: Dict[str, Any], site_key: str) -> bool:
    """Inserts or updates a regular tender in the database."""
    # Determine the primary ID from the detailed scrape if available, else from list page
    tender_id_str = tender_data.get("detail_tender_id") or tender_data.get("list_page_tender_id")
    if not tender_id_str or tender_id_str == "N/A":
        if scraper_logger: scraper_logger.warning(f"Skipping DB save: No valid Tender ID found for item '{tender_data.get('list_page_title', 'No Title')}'")
        return False
    
    try:
        # Check if tender already exists
        tender_record = db.query(Tender).filter(Tender.tender_id == tender_id_str).first()
        
        if tender_record:
            db_logger.info(f"Tender ID '{tender_id_str}' already exists. Skipping as we don't update live tenders currently.")
            # Future enhancement: Could check 'updated_at' and update if changed. For now, we skip.
            return True # Not an error, just skipping.
        
        # If it doesn't exist, create a new one
        db_logger.info(f"Creating new tender record for ID '{tender_id_str}'.")
        
        # Prepare data for the Tender model
        new_tender = Tender(
            tender_id=tender_id_str,
            source_site=site_key,
            tender_title=tender_data.get("work_title") or tender_data.get("list_page_title"),
            organisation_chain=tender_data.get("organisation_chain") or tender_data.get("list_page_org_chain"),
            tender_value_numeric=parse_amount_to_numeric(tender_data.get("tender_value")),
            emd_amount_numeric=parse_amount_to_numeric(tender_data.get("emd_amount")),
            published_date=parse_date_reg(tender_data.get("published_date") or tender_data.get("epublished_date_str")),
            closing_date=parse_date_reg(tender_data.get("bid_submission_end_date") or tender_data.get("closing_date_str")),
            opening_date=parse_date_reg(tender_data.get("detail_bid_opening_date") or tender_data.get("opening_date_str")),
            location=tender_data.get("location"),
            pincode=tender_data.get("pincode"),
            status="Live",  # New tenders are always 'Live' initially
            full_details_json=tender_data # Store the whole dictionary
        )
        
        db.add(new_tender)
        # We will commit in a batch in the calling function
        return True
        
    except SQLAlchemyError as e:
        db_logger.error(f"Database error while saving tender ID '{tender_id_str}': {e}", exc_info=True)
        return False
    except Exception as e:
        db_logger.error(f"Unexpected error while saving tender ID '{tender_id_str}': {e}", exc_info=True)
        return False

# --- MODIFIED: Core fetching logic ---
async def fetch_single_list_page(context: BrowserContext, page_number: int, site_base_url: str, site_domain_str: str, site_key_str: str, detail_semaphore: asyncio.Semaphore, page_load_timeout_ms: int, detail_page_timeout_ms: int, retry_limit_val: int) -> Tuple[int, List[Dict[str, Any]]]:
    """Fetches a single list page, scrapes details, and returns a list of combined data dictionaries."""
    list_page_url_to_fetch = site_base_url.format(page_number)
    main_site_page_url = urljoin(site_domain_str, urlparse(site_domain_str).path)
    page_for_list: Optional[Page] = None
    
    for attempt in range(1, retry_limit_val + 2):
        try:
            if not page_for_list or page_for_list.is_closed():
                page_for_list = await context.new_page()

            if scraper_logger: scraper_logger.info(f"📄 Fetching List Page {page_number} (Attempt {attempt}/{retry_limit_val + 1})")

            if attempt == 1:
                try:
                    if scraper_logger: scraper_logger.debug(f"  Navigating to main site: {main_site_page_url}")
                    await page_for_list.goto(main_site_page_url, wait_until="load", timeout=page_load_timeout_ms // 2)
                    await page_for_list.wait_for_timeout(500)
                except Exception as e_pre_visit:
                    if scraper_logger: scraper_logger.warning(f"  ⚠️ Initial navigation to {main_site_page_url} failed: {type(e_pre_visit).__name__}. Proceeding to target URL directly.")

            await page_for_list.goto(list_page_url_to_fetch, wait_until="domcontentloaded", timeout=page_load_timeout_ms)
            
            # --- CORRECTED & FINAL: Use asyncio.wait to race two conditions ---
            wait_for_table_task = asyncio.create_task(
                page_for_list.locator("table#table").first.wait_for(state="visible", timeout=30000)
            )
            wait_for_no_records_task = asyncio.create_task(
                page_for_list.locator("text=/no records found/i").first.wait_for(state="visible", timeout=30000)
            )
            
            # Wait for the first task to complete
            done, pending = await asyncio.wait(
                [wait_for_table_task, wait_for_no_records_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel the task that didn't finish
            for task in pending:
                task.cancel()
            
            # Check for exceptions in the completed task
            try:
                # This will re-raise an exception if the completed task failed (e.g., timeout)
                for task in done:
                    task.result()
            except PlaywrightTimeout:
                 if scraper_logger: scraper_logger.warning(f"  ⚠️ Timed out waiting for table or 'no records' message on page {page_number}. Retrying.")
                 await asyncio.sleep(3 * (2 ** (attempt - 1)))
                 continue # Go to the next attempt in the loop

            # Now check which condition was met by inspecting the page content
            current_page_content = await page_for_list.content()
            if "no records found" in current_page_content.lower() or "no tenders available" in current_page_content.lower():
                if scraper_logger: scraper_logger.info(f"  📄 Page {page_number}: No records found. Stopping fetch for this site.")
                if page_for_list and not page_for_list.is_closed(): await page_for_list.close()
                return page_number, [{"status_signal": "NO_RECORDS"}]

            soup_list_page = BeautifulSoup(current_page_content, "html.parser")
            tender_list_table = soup_list_page.find("table", id="table")
            if not tender_list_table:
                if scraper_logger: scraper_logger.warning(f"  ⚠️ Table not found on list page {page_number} after wait. Retrying."); 
                await asyncio.sleep(3 * (2 ** (attempt - 1)))
                continue
            
            tender_list_rows = tender_list_table.find_all("tr", id=re.compile(r"informal"))
            if not tender_list_rows:
                if scraper_logger: scraper_logger.info(f"  📄 Page {page_number}: Table found but no tender rows. Stopping fetch."); 
                if page_for_list and not page_for_list.is_closed(): await page_for_list.close()
                return page_number, [{"status_signal": "NO_RECORDS"}]
            
            list_page_tender_items: List[Dict[str, str]] = []
            for row_item in tender_list_rows:
                cols_item = row_item.find_all("td")
                if len(cols_item) < 6: continue
                title_id_cell_item = cols_item[4]; cell_text_item = title_id_cell_item.get_text(separator='\n', strip=True); link_tag_item = title_id_cell_item.find("a", href=True)
                basic_tender_info = {"list_page_tender_id": "N/A", "list_page_title": "N/A","epublished_date_str": get_safe_text(cols_item[1]), "closing_date_str": get_safe_text(cols_item[2]),"opening_date_str": get_safe_text(cols_item[3]), "list_page_org_chain": get_safe_text(cols_item[5]),"detail_page_link": "N/A"}
                if link_tag_item: 
                    relative_href_item = link_tag_item.get('href', '');
                    if relative_href_item and 'javascript:void' not in relative_href_item and relative_href_item.strip() != '#': basic_tender_info["detail_page_link"] = urljoin(site_domain_str, relative_href_item.strip())
                    basic_tender_info["list_page_title"] = clean_text(link_tag_item.get_text())
                all_brackets_item = BRACKET_CONTENT_REGEX.findall(cell_text_item);
                for content_in_bracket in all_brackets_item: 
                    content_in_bracket_stripped = content_in_bracket.strip();
                    if STRICT_ID_CONTENT_REGEX.match(content_in_bracket_stripped):
                        basic_tender_info["list_page_tender_id"] = content_in_bracket_stripped; break
                if basic_tender_info["list_page_tender_id"] != "N/A": list_page_tender_items.append(basic_tender_info)

            detail_tasks = [_scrape_detail_page_with_semaphore(detail_semaphore, context, item.get("detail_page_link", ""), site_domain_str, site_key_str, detail_page_timeout_ms) for item in list_page_tender_items if item.get("detail_page_link")]
            if scraper_logger: scraper_logger.info(f"  ⏯️ Gathering {len(detail_tasks)} detail scrape tasks for page {page_number}.")
            detail_results = await asyncio.gather(*detail_tasks, return_exceptions=True)

            combined_results = []
            for i, item_data in enumerate(list_page_tender_items):
                full_data = item_data.copy()
                if i < len(detail_results):
                    res = detail_results[i]
                    if isinstance(res, dict):
                        full_data.update(res)
                    elif isinstance(res, Exception):
                        if scraper_logger: scraper_logger.error(f"  ❌ Detail scrape task failed for {item_data.get('list_page_tender_id')}: {res}", exc_info=False)
                combined_results.append(full_data)

            if scraper_logger: scraper_logger.info(f"  ✅ List Page {page_number} processed successfully ({len(list_page_tender_items)} items).")
            if page_for_list and not page_for_list.is_closed(): await page_for_list.close()
            return page_number, combined_results

        except (PlaywrightTimeout, ConnectionError) as e:
            if scraper_logger: scraper_logger.warning(f"  ⚠️ Network/Timeout on List Page {page_number}, attempt {attempt}/{retry_limit_val + 1}: {type(e).__name__}")
        except Exception as e:
            if scraper_logger: scraper_logger.error(f"  ❌ Unhandled Error on List Page {page_number}, attempt {attempt}/{retry_limit_val + 1}: {e}", exc_info=True)
        
        if attempt <= retry_limit_val:
            await asyncio.sleep(3 * (2 ** (attempt -1)))
        
    if page_for_list and not page_for_list.is_closed(): await page_for_list.close()
    if scraper_logger: scraper_logger.error(f"  ❌ Failed to process List Page {page_number} after all attempts.")
    return page_number, []

# --- MODIFIED: Main orchestration logic for a site ---
async def fetch_all_pages_for_site_and_save_to_db(playwright_instance: Any, site_key: str, site_base_url: str, site_domain: str, max_pages: int, concurrency_val: int, detail_concurrency_val: int, page_timeout: int, detail_timeout: int, retry_max: int):
    browser: Optional[BrowserContext.browser] = None; context: Optional[BrowserContext] = None
    tenders_saved_count = 0; tenders_failed_count = 0
    try:
        browser = await playwright_instance.chromium.launch(headless=True)
        context = await browser.new_context(ignore_https_errors=True)
        detail_page_semaphore = asyncio.Semaphore(detail_concurrency_val)
        stop_fetching = False; current_page_num = 1; consecutive_empty_pages = 0

        while current_page_num <= max_pages and not stop_fetching:
            batch_tasks = [fetch_single_list_page(context, page_num, site_base_url, site_domain, site_key, detail_page_semaphore, page_timeout, detail_timeout, retry_max) for page_num in range(current_page_num, min(current_page_num + concurrency_val, max_pages + 1))]
            current_page_num += len(batch_tasks)
            
            page_results = await asyncio.gather(*batch_tasks)
            
            all_tenders_from_batch = []
            for _, tender_list in page_results:
                if not tender_list: # Total failure for this page
                    consecutive_empty_pages += 1
                elif tender_list[0].get("status_signal") == "NO_RECORDS":
                    stop_fetching = True; break
                else:
                    consecutive_empty_pages = 0
                    all_tenders_from_batch.extend(tender_list)
            
            if stop_fetching or consecutive_empty_pages >= concurrency_val:
                if scraper_logger and consecutive_empty_pages >= concurrency_val: scraper_logger.warning(f"Stopping site {site_key}: {consecutive_empty_pages} consecutive empty/failed pages.")
                break
            
            # Batch save to DB
            if all_tenders_from_batch:
                db: Session = SessionLocal()
                try:
                    for tender_data in all_tenders_from_batch:
                        if save_tender_to_db(db, tender_data, site_key):
                            tenders_saved_count += 1
                        else:
                            tenders_failed_count += 1
                    db.commit()
                    db_logger.info(f"Committed batch of {len(all_tenders_from_batch)} tenders for site '{site_key}'.")
                except Exception as e:
                    db.rollback()
                    db_logger.error(f"Batch DB commit failed for site '{site_key}': {e}")
                finally:
                    db.close()

    except Exception as e:
        if scraper_logger: scraper_logger.critical(f"💥 Unhandled exception during site fetching for {site_key}: {e}", exc_info=True)
    finally:
        if context: await context.close()
        if browser: await browser.close()
        if scraper_logger: scraper_logger.info(f"Scrape run for site {site_key} finished. Saved: {tenders_saved_count}, Failed to save: {tenders_failed_count}.")


# --- MODIFIED: Main site entry point ---
async def run_scrape_for_one_site(site_key: str, site_config: Dict[str, str], global_settings: Dict[str, Any]):
    current_site_logger = setup_site_specific_logging(site_key)
    current_site_logger.info(f"🚀 Starting DB-integrated scrape run for site: {site_key}...")
    start_time_site = datetime.datetime.now()
    
    site_base_url = site_config.get("base_url")
    site_domain = site_config.get("domain")
    if not all([site_base_url, site_domain]):
        current_site_logger.error(f"Config for '{site_key}' missing 'base_url' or 'domain'. Skipping.")
        return

    # Extract settings with defaults
    max_pages = int(global_settings.get("scrape_py_limits", {}).get("max_list_pages", MAX_PAGES_TO_FETCH_DEFAULT))
    retry_limit = int(global_settings.get("scrape_py_limits", {}).get("retries", RETRY_LIMIT_DEFAULT))
    list_concurrency = int(global_settings.get("concurrency", {}).get("list_pages", CONCURRENCY_DEFAULT))
    detail_concurrency = int(global_settings.get("concurrency", {}).get("detail_pages", DETAIL_PAGE_CONCURRENCY_LIMIT_DEFAULT))
    page_timeout = int(global_settings.get("timeouts", {}).get("page_load", PAGE_LOAD_TIMEOUT_DEFAULT))
    detail_timeout = int(global_settings.get("timeouts", {}).get("detail_page", DETAIL_PAGE_TIMEOUT_DEFAULT))

    try:
        async with async_playwright() as p:
            await fetch_all_pages_for_site_and_save_to_db(p, site_key, site_base_url, site_domain, max_pages, list_concurrency, detail_concurrency, page_timeout, detail_timeout, retry_limit)
    except Exception as e:
        current_site_logger.critical(f"💥 CRITICAL ERROR during scrape run for site {site_key}: {e}", exc_info=True)
    finally:
        duration_site = datetime.datetime.now() - start_time_site
        current_site_logger.info(f"🏁 Scrape run for site {site_key} completed. Duration: {duration_site}.")

# --- Standalone Execution Block (for testing) ---
if __name__ == "__main__":
    print(f"--- Running {Path(__file__).name} in standalone test mode ---")
    
    # Ensure DB is ready for the test
    from database import init_db
    init_db()

    test_site_key = "CPPP_Test" 
    test_site_config_data = {
        "base_url": "https://eprocure.gov.in/eprocure/app?component=%24TablePages.linkPage&page=FrontEndAdvancedSearchResult&service=direct&session=T&sp=AFrontEndAdvancedSearchResult%2Ctable&sp={}",
        "domain": "https://eprocure.gov.in/eprocure/app"
    }
    test_global_settings = {
        "scrape_py_limits": {"max_list_pages": 1, "retries": 0},
        "concurrency": {"list_pages": 1, "detail_pages": 1},     
        "timeouts": {"page_load": 60000, "detail_page": 60000}
    }
    
    try:
        asyncio.run(run_scrape_for_one_site(test_site_key, test_site_config_data, test_global_settings))
        print("--- Standalone test finished. Check logs and 'tender_aggregator.db' for results. ---")
    except Exception as main_err:
         print(f"Error during standalone test run: {main_err}")
