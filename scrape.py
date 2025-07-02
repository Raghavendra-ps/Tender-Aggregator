# Tender-Aggregator/scrape.py

import asyncio
import datetime
import logging
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
from urllib.parse import urljoin
import random

from bs4 import BeautifulSoup, Tag
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout, Page, BrowserContext, Error as PlaywrightError

# --- Database Imports ---
import sys
if str(Path(__file__).parent.resolve()) not in sys.path:
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
RETRY_LIMIT_DEFAULT = 2
CONCURRENCY_DEFAULT = 4
DETAIL_PAGE_CONCURRENCY_LIMIT_DEFAULT = 2
PAGE_LOAD_TIMEOUT_DEFAULT = 90000
DETAIL_PAGE_TIMEOUT_DEFAULT = 90000

# --- Directory Constants & Logging Setup ---
PROJECT_ROOT = Path(__file__).parent.resolve()
LOGS_BASE_DIR = PROJECT_ROOT / "LOGS"
scraper_logger: Optional[logging.Logger] = None

def setup_site_specific_logging(site_key: str) -> logging.Logger:
    global scraper_logger
    regular_scraper_log_dir = LOGS_BASE_DIR / "regular_scraper"
    regular_scraper_log_dir.mkdir(parents=True, exist_ok=True)
    safe_site_key_log = re.sub(r'[^\w\-]+', '_', site_key)
    log_file_path = regular_scraper_log_dir / f"scrape_{safe_site_key_log}.log"
    logger_name = f'scraper_regular_{site_key}'
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(f"%(asctime)s [%(levelname)s] (Scraper-REG-{site_key}) %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(f"[%(levelname)s - Scraper-REG-{site_key}] %(message)s"))
        logger.addHandler(sh)
    scraper_logger = logger
    return logger

# --- Helper Functions ---
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
            caption_pattern = re.compile(r'^\s*' + re.escape(core_caption) + r'\s*:?\s*$', re.IGNORECASE)
            caption_td = container.find(lambda tag: tag.name == 'td' and 'td_caption' in tag.get('class', []) and caption_pattern.search(re.sub(r'\s*\(.*?\)\s*', '', tag.get_text(strip=True)).strip()))
            if caption_td and (value_td := caption_td.find_next_sibling('td')):
                return clean_text(value_td.get_text())
        except Exception: pass
    return "N/A"
def parse_date_reg(date_str: Optional[str]) -> Optional[datetime.datetime]:
    if not date_str or not isinstance(date_str, str): return None
    formats = ['%d-%b-%Y %I:%M %p', '%d-%b-%Y'];
    for fmt in formats:
        try: return datetime.datetime.strptime(date_str, fmt)
        except ValueError: continue
    return None
def parse_amount_to_numeric(amount_text: Optional[str]) -> Optional[float]:
    if not amount_text or not isinstance(amount_text, str): return None
    try:
        cleaned_text = re.sub(r'[^\d\.\-]', '', amount_text)
        if cleaned_text and (cleaned_text.replace('.', '', 1).replace('-', '', 1).isdigit() or (cleaned_text.startswith('-') and cleaned_text[1:].replace('.', '', 1).isdigit())):
            return float(cleaned_text)
    except (ValueError, TypeError): pass
    return None

# --- Detail Page Scraping ---
async def _scrape_detail_page(page: Page, detail_url: str, detail_page_timeout_ms: int) -> Optional[Dict[str, Any]]:
    try:
        if scraper_logger: scraper_logger.debug(f"    ➡️  Scraping Detail Page: {detail_url}")
        await page.goto(detail_url, wait_until="domcontentloaded", timeout=detail_page_timeout_ms)
        html = await page.content()
        if not html:
            if scraper_logger: scraper_logger.warning(f"    ⚠️ No content for detail page: {detail_url}")
            return None
        soup = BeautifulSoup(html, 'html.parser')
        details: Dict[str, Any] = {}
        def find_data_table_after_header(header_text_variants: List[str]) -> Optional[Tag]:
            for header_text in header_text_variants:
                header_td = soup.find(lambda tag: tag.name == "td" and header_text.lower() in tag.get_text(strip=True).lower() and "pageheader" in tag.get('class', []))
                if header_td and (parent_row := header_td.find_parent('tr')) and (data_row := parent_row.find_next_sibling('tr')) and data_row.find('td'):
                    return data_row.find('td').find('table', class_=re.compile(r'(tablebg|list_table)'))
            return None
        if basic_details_table := find_data_table_after_header(["Basic Details"]):
            details.update({"organisation_chain": _find_detail_field_value(basic_details_table, "Organisation Chain"), "tender_reference_number": _find_detail_field_value(basic_details_table, "Tender Reference Number"), "detail_tender_id": _find_detail_field_value(basic_details_table, "Tender ID"), "tender_type": _find_detail_field_value(basic_details_table, "Tender Type"), "tender_category": _find_detail_field_value(basic_details_table, "Tender Category"), "form_of_contract": _find_detail_field_value(basic_details_table, "Form Of Contract")})
        if work_item_table := find_data_table_after_header(["Work Item Details"]):
            details.update({"work_title": _find_detail_field_value(work_item_table, "Title"), "work_description": _find_detail_field_value(work_item_table, "Work Description"), "tender_value": _find_detail_field_value(work_item_table, "Tender Value in ₹"), "location": _find_detail_field_value(work_item_table, "Location"), "pincode": _find_detail_field_value(work_item_table, "Pincode")})
        if critical_dates_table := find_data_table_after_header(["Critical Dates"]):
            details.update({"published_date": _find_detail_field_value(critical_dates_table, "Published Date"), "closing_date": _find_detail_field_value(critical_dates_table, "Bid Submission End Date"), "opening_date": _find_detail_field_value(critical_dates_table, "Bid Opening Date")})
        return details
    except PlaywrightError as e:
        if scraper_logger: scraper_logger.warning(f"    ⚠️ Playwright error on detail page {detail_url}: {type(e).__name__}")
    except Exception as e:
        if scraper_logger: scraper_logger.error(f"    ❌ Unhandled error on detail page {detail_url}: {e}", exc_info=False)
    return None

async def _scrape_detail_page_with_semaphore(semaphore: asyncio.Semaphore, context: BrowserContext, list_page_item: Dict, detail_page_timeout_ms: int) -> Dict[str, Any]:
    async with semaphore:
        await asyncio.sleep(random.uniform(0.2, 0.8))
        page: Optional[Page] = None
        detail_url = list_page_item.get("detail_page_link")
        full_tender_data = list_page_item.copy()
        if not detail_url or detail_url == "N/A":
            return full_tender_data
        try:
            page = await context.new_page()
            detail_data = await _scrape_detail_page(page, detail_url, detail_page_timeout_ms)
            if detail_data:
                full_tender_data.update(detail_data)
        except Exception as e:
            if scraper_logger: scraper_logger.error(f"Error in detail page task for {detail_url}: {e}")
        finally:
            if page and not page.is_closed(): await page.close()
        return full_tender_data

# --- Database Interaction ---
def save_tender_to_db(db: Session, tender_data: Dict[str, Any], site_key: str):
    tender_id_str = tender_data.get("detail_tender_id") or tender_data.get("list_page_tender_id")
    if not tender_id_str or tender_id_str == "N/A":
        return
    
    try:
        tender_record = db.query(Tender).filter(Tender.tender_id == tender_id_str, Tender.source_site == site_key).first()
        if not tender_record:
            tender_record = Tender(tender_id=tender_id_str, source_site=site_key)
            db.add(tender_record)
        
        tender_record.tender_title = tender_data.get("work_title") or tender_data.get("list_page_title")
        tender_record.organisation_chain = tender_data.get("organisation_chain") or tender_data.get("list_page_org_chain")
        tender_record.tender_value_numeric = parse_amount_to_numeric(tender_data.get("tender_value"))
        tender_record.published_date = parse_date_reg(tender_data.get("published_date") or tender_data.get("epublished_date_str"))
        tender_record.closing_date = parse_date_reg(tender_data.get("closing_date") or tender_data.get("closing_date_str"))
        tender_record.opening_date = parse_date_reg(tender_data.get("opening_date") or tender_data.get("opening_date_str"))
        tender_record.location = tender_data.get("location")
        tender_record.pincode = tender_data.get("pincode")
        tender_record.full_details_json = tender_data
        db.commit() # Commit each record individually
    except Exception as e:
        db.rollback()
        db_logger.error(f"DB error for tender '{tender_id_str}': {e}")

# --- Core Scraping Task ---
async def fetch_and_process_page(
    context: BrowserContext, 
    page_number: int, 
    site_config: dict, 
    timeouts: dict,
    retry_limit: int, 
    list_semaphore: asyncio.Semaphore,
    detail_semaphore: asyncio.Semaphore
):
    async with list_semaphore:
        list_page_url = site_config['base_url'].format(page_number)
        for attempt in range(1, retry_limit + 2):
            page: Optional[Page] = None
            try:
                page = await context.new_page()
                if scraper_logger: scraper_logger.info(f"📄 Fetching List Page {page_number} (Attempt {attempt})...")
                
                await page.goto(list_page_url, wait_until="domcontentloaded", timeout=timeouts.get('page_load', PAGE_LOAD_TIMEOUT_DEFAULT))
                
                html = await page.content()
                soup = BeautifulSoup(html, "html.parser")
                
                if "no records found" in soup.body.get_text(strip=True).lower():
                    if scraper_logger: scraper_logger.info(f"  ✅ Page {page_number}: 'No records' found. This is the last page.")
                    await page.close()
                    return 'STOP' # Signal to stop

                rows = soup.select("table#table tr[id^='informal']")
                if not rows:
                    if scraper_logger: scraper_logger.warning(f"  ⚠️ Page {page_number}: No tender rows found. Assuming end.")
                    await page.close()
                    return 'STOP'
                
                list_items = []
                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) < 6: continue
                    link_tag = cols[4].find("a", href=True)
                    tender_id = next((c.strip() for c in re.findall(r"\[(.*?)\]", cols[4].get_text()) if re.match(r"^\d{4}_\w+_\d+_\d+$", c.strip())), "N/A")
                    list_items.append({
                        "list_page_title": clean_text(link_tag.get_text()) if link_tag else "N/A",
                        "epublished_date_str": get_safe_text(cols[1]), "closing_date_str": get_safe_text(cols[2]),
                        "opening_date_str": get_safe_text(cols[3]), "list_page_org_chain": get_safe_text(cols[5]),
                        "detail_page_link": urljoin(site_config['domain'], link_tag["href"]) if link_tag else "N/A",
                        "list_page_tender_id": tender_id
                    })
                await page.close()

                # Process details for this page's items
                detail_tasks = [
                    scrape_and_save_details(context, item, site_config, timeouts, detail_semaphore)
                    for item in list_items
                ]
                await asyncio.gather(*detail_tasks)
                return 'CONTINUE' # Signal to continue

            except PlaywrightError as e:
                if scraper_logger: scraper_logger.error(f"  ❌ Playwright Error on Page {page_number}, attempt {attempt}: {e.message.splitlines()[0]}")
            except Exception as e:
                if scraper_logger: scraper_logger.error(f"  ❌ Unhandled Error on Page {page_number}, attempt {attempt}: {e}", exc_info=True)
            finally:
                if page and not page.is_closed(): await page.close()
            if attempt <= retry_limit: await asyncio.sleep(3 * attempt)
        
        return 'FAIL' # Signal that this page failed after all retries

async def scrape_and_save_details(
    context: BrowserContext, 
    item: dict, 
    site_config: dict, 
    timeouts: dict,
    detail_semaphore: asyncio.Semaphore
):
    async with detail_semaphore:
        page: Optional[Page] = None
        try:
            if item['detail_page_link'] == 'N/A': return
            page = await context.new_page()
            details = await _scrape_detail_page(page, item['detail_page_link'], timeouts.get('detail_page', DETAIL_PAGE_TIMEOUT_DEFAULT))
            if details:
                item.update(details)
            
            db = SessionLocal()
            try:
                save_tender_to_db(db, item, site_config['key'])
            finally:
                db.close()
        except Exception as e:
            if scraper_logger: scraper_logger.error(f"  ❌ Error in detail processing for {item.get('detail_page_link')}: {e}")
        finally:
            if page and not page.is_closed(): await page.close()


# --- Main Orchestration ---
async def run_scrape_for_one_site(site_key: str, site_config: Dict[str, str], global_settings: Dict[str, Any]):
    logger = setup_site_specific_logging(site_key)
    logger.info(f"🚀 Starting DB-integrated scrape run for site: {site_key}...")
    start_time = datetime.datetime.now()
    
    site_config['key'] = site_key # Add key to config for easy access
    timeouts = global_settings.get("timeouts", {})
    limits = global_settings.get("scrape_py_limits", {})
    concurrency = global_settings.get("concurrency", {})
    retry_max = limits.get('retries', RETRY_LIMIT_DEFAULT)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(ignore_https_errors=True)
        try:
            # Proactively handle popups by visiting the homepage once
            logger.info("Performing initial homepage visit to handle popups...")
            page = await context.new_page()
            try:
                await page.goto(site_config['domain'], wait_until="domcontentloaded", timeout=timeouts.get('page_load', PAGE_LOAD_TIMEOUT_DEFAULT))
                await page.locator("div#msgbox button:has-text('Close')").click(timeout=7000)
                logger.info("  'Important Message' popup closed on initial visit.")
            except PlaywrightError:
                logger.info("  No initial popup found, or it closed quickly.")
            finally:
                await page.close()

            list_semaphore = asyncio.Semaphore(concurrency.get('list_pages', CONCURRENCY_DEFAULT))
            detail_semaphore = asyncio.Semaphore(concurrency.get('detail_pages', DETAIL_PAGE_CONCURRENCY_LIMIT_DEFAULT))
            
            tasks = []
            for page_num in range(1, limits.get('max_list_pages', MAX_PAGES_TO_FETCH_DEFAULT) + 1):
                task = asyncio.create_task(
                    fetch_and_process_page(context, page_num, site_config, timeouts, retry_max, list_semaphore, detail_semaphore)
                )
                tasks.append(task)

            # Wait for tasks to complete, and stop if any signal the end
            for future in asyncio.as_completed(tasks):
                result = await future
                if result == 'STOP':
                    logger.info("Stop signal received, cancelling remaining page fetch tasks.")
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    break
        finally:
            await context.close()
            await browser.close()

    duration = datetime.datetime.now() - start_time
    logger.info(f"🏁 Scrape run for site {site_key} completed. Duration: {duration}.")

# --- Standalone Execution Block (for testing) ---
if __name__ == "__main__":
    from database import init_db
    init_db()
    # ... (standalone testing logic can be added here)
