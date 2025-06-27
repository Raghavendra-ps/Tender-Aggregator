# Tender-Aggregator/scrape.py

import asyncio
import datetime
import logging
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
from urllib.parse import urljoin, urlparse
import random

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
CONCURRENCY_DEFAULT = 4 
DETAIL_PAGE_CONCURRENCY_LIMIT_DEFAULT = 2
PAGE_LOAD_TIMEOUT_DEFAULT = 75000
DETAIL_PAGE_TIMEOUT_DEFAULT = 75000

# --- Directory Constants (LOGS ONLY) ---
PROJECT_ROOT = Path(__file__).parent.resolve()
LOGS_BASE_DIR = PROJECT_ROOT / "LOGS"

# === LOG CONFIG ===
scraper_logger: Optional[logging.Logger] = None

def setup_site_specific_logging(site_key: str) -> logging.Logger:
    global scraper_logger
    regular_scraper_log_dir = LOGS_BASE_DIR / "regular_scraper"
    regular_scraper_log_dir.mkdir(parents=True, exist_ok=True)
    safe_site_key_log = re.sub(r'[^\w\-]+', '_', site_key)
    log_file_path = regular_scraper_log_dir / f"scrape_{safe_site_key_log}.log"
    logger_name = f'scraper_regular_{site_key}'
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        scraper_logger = logger
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(f"%(asctime)s [%(levelname)s] (Scraper-REG-{site_key}) %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(fh)
    except Exception as e:
        print(f"CRITICAL: Failed to initialize file log handler for {log_file_path}: {e}")
    sh = logging.StreamHandler()
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
            if caption_td:
                value_td = caption_td.find_next_sibling('td', class_='td_field')
                if not value_td: value_td = caption_td.find_next_sibling('td')
                if value_td: return clean_text(value_td.get_text())
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
    if not amount_text or not isinstance(amount_text, str): return None
    try:
        cleaned_text = re.sub(r'[^\d\.\-]', '', amount_text)
        if cleaned_text and (cleaned_text.replace('.', '', 1).replace('-', '', 1).isdigit() or (cleaned_text.startswith('-') and cleaned_text[1:].replace('.', '', 1).isdigit())):
            return float(cleaned_text)
    except (ValueError, TypeError): pass
    if scraper_logger: scraper_logger.debug(f"Could not parse amount: '{amount_text}' to float.")
    return None

# --- Detail Page Scraping ---

async def _scrape_detail_page(page: Page, detail_url: str, site_domain: str, site_key: str, detail_page_timeout_ms: int) -> Optional[Dict[str, Any]]:
    try:
        if scraper_logger: scraper_logger.info(f"    ➡️  Scraping Detail Page: {detail_url}")
        await page.goto(detail_url, wait_until="domcontentloaded", timeout=detail_page_timeout_ms)
        html = await page.content()
        if not html:
            if scraper_logger: scraper_logger.warning(f"    ⚠️ No content received for detail page: {detail_url}")
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        details: Dict[str, Any] = {
            "organisation_chain": "N/A", "tender_reference_number": "N/A", "detail_tender_id": "N/A", "withdrawal_allowed": "N/A", "tender_type": "N/A", "form_of_contract": "N/A", "tender_category": "N/A", "no_of_covers": "N/A", "general_tech_evaluation": "N/A", "itemwise_tech_evaluation": "N/A", "payment_mode": "N/A", "multi_currency_boq": "N/A", "multi_currency_fee": "N/A", "two_stage_bidding": "N/A", "payment_instruments": [], "covers_info": [], "tender_fee": "N/A", "processing_fee": "N/A", "fee_payable_to": "N/A", "fee_payable_at": "N/A", "tender_fee_exemption": "N/A", "emd_amount": "N/A", "emd_exemption": "N/A", "emd_fee_type": "N/A", "emd_percentage": "N/A", "emd_payable_to": "N/A", "emd_payable_at": "N/A", "work_title": "N/A", "work_description": "N/A", "nda_pre_qualification": "N/A", "independent_external_monitor_remarks": "N/A", "tender_value": "N/A", "product_category": "N/A", "sub_category": "N/A", "contract_type": "N/A", "bid_validity_days": "N/A", "period_of_work_days": "N/A", "location": "N/A", "pincode": "N/A", "pre_bid_meeting_place": "N/A", "pre_bid_meeting_address": "N/A", "pre_bid_meeting_date": "N/A", "bid_opening_place": "N/A", "allow_nda_tender": "N/A", "allow_preferential_bidder": "N/A", "published_date": "N/A", "detail_bid_opening_date": "N/A", "doc_download_start_date": "N/A", "doc_download_end_date": "N/A", "clarification_start_date": "N/A", "clarification_end_date": "N/A", "bid_submission_start_date": "N/A", "bid_submission_end_date": "N/A", "tender_documents": [], "tender_inviting_authority_name": "N/A", "tender_inviting_authority_address": "N/A"
        }
        
        def find_data_table_after_header(header_text_variants: List[str]) -> Optional[Tag]:
            for header_text in header_text_variants:
                header_td = soup.find(lambda tag: tag.name == "td" and header_text.lower() in tag.get_text(strip=True).lower() and "pageheader" in tag.get('class', []))
                if header_td:
                    parent_row = header_td.find_parent('tr')
                    if parent_row:
                        data_row = parent_row.find_next_sibling('tr')
                        if data_row and data_row.find('td'):
                            return data_row.find('td').find('table', class_=re.compile(r'(tablebg|list_table)'))
            return None

        # --- Section: Basic Details ---
        basic_details_table = find_data_table_after_header(["Basic Details"])
        if basic_details_table:
            details["organisation_chain"] = _find_detail_field_value(basic_details_table, "Organisation Chain")
            details["tender_reference_number"] = _find_detail_field_value(basic_details_table, "Tender Reference Number")
            details["detail_tender_id"] = _find_detail_field_value(basic_details_table, "Tender ID")
            details["tender_type"] = _find_detail_field_value(basic_details_table, "Tender Type")
            details["tender_category"] = _find_detail_field_value(basic_details_table, "Tender Category")
            details["general_tech_evaluation"] = _find_detail_field_value(basic_details_table, "General Technical Evaluation Allowed")
            details["payment_mode"] = _find_detail_field_value(basic_details_table, "Payment Mode")
            details["withdrawal_allowed"] = _find_detail_field_value(basic_details_table, "Withdrawal of Bid Allowed")
            details["form_of_contract"] = _find_detail_field_value(basic_details_table, "Form Of Contract")
            details["no_of_covers"] = _find_detail_field_value(basic_details_table, "No. of Covers")
            details["itemwise_tech_evaluation"] = _find_detail_field_value(basic_details_table, "ItemWise Technical Evaluation Allowed")
            details["multi_currency_boq"] = _find_detail_field_value(basic_details_table, "Is Multi Currency Allowed For BOQ")
            details["allow_two_stage_bidding"] = _find_detail_field_value(basic_details_table, "Allow Two Stage Bidding")

        # --- Section: Payment Instruments ---
        payment_instruments_table = find_data_table_after_header(["Payment Instruments"])
        if payment_instruments_table and payment_instruments_table.find('table'):
            pi_list = []
            rows = payment_instruments_table.find('table').find_all('tr')
            for row in rows:
                if row.find('th') or 'list_header' in row.get('class', []): continue
                cols = row.find_all('td')
                if len(cols) >= 2:
                    s_no = get_safe_text(cols[0])
                    instrument_type = get_safe_text(cols[1])
                    if s_no.lower() in ["s.no", "s.no."] or not s_no.strip() or s_no == "N/A": continue
                    pi_list.append({"s_no": s_no, "instrument_type": instrument_type})
            details["payment_instruments"] = pi_list

        # --- Section: Covers Information ---
        covers_table = find_data_table_after_header(["Covers Information", "Cover Details"])
        if covers_table and covers_table.find('table'):
            ci_list = []
            rows = covers_table.find('table').find_all('tr')
            for row in rows:
                if row.find('th') or 'list_header' in row.get('class', []): continue
                cols = row.find_all('td')
                if len(cols) >= 2:
                    cover_no = get_safe_text(cols[0])
                    cover_type = get_safe_text(cols[1])
                    if cover_no.lower() in ["cover no", "cover no."] or not cover_no.strip() or cover_no == "N/A": continue
                    ci_list.append({"cover_no": cover_no, "cover_type": cover_type})
            details["covers_info"] = ci_list
        
        # --- Section: Tender Fee Details & EMD Fee Details ---
        fee_details_header = soup.find('td', class_='pageheader', string=re.compile(r"Tender Fee Details"))
        if fee_details_header:
            fee_table = fee_details_header.find_parent("tr").find_next_sibling("tr").find("table", class_="tablebg")
            if fee_table:
                details["tender_fee"] = _find_detail_field_value(fee_table, "Tender Fee in ₹")
                details["fee_payable_to"] = _find_detail_field_value(fee_table, "Fee Payable To")
                details["fee_payable_at"] = _find_detail_field_value(fee_table, "Fee Payable At")
                details["tender_fee_exemption"] = _find_detail_field_value(fee_table, "Tender Fee Exemption Allowed")
        
        emd_details_header = soup.find('td', class_='pageheader', string=re.compile(r"EMD Fee Details"))
        if emd_details_header:
            emd_table = emd_details_header.find_parent("tr").find_next_sibling("tr").find("table", class_="tablebg")
            if emd_table:
                details["emd_amount"] = _find_detail_field_value(emd_table, "EMD Amount in ₹")
                details["emd_fee_type"] = _find_detail_field_value(emd_table, "EMD Fee Type")
                details["emd_payable_to"] = _find_detail_field_value(emd_table, "EMD Payable To")
                details["emd_payable_at"] = _find_detail_field_value(emd_table, "EMD Payable At")

        # --- Section: Work Item Details ---
        work_item_table = find_data_table_after_header(["Work Item Details"])
        if work_item_table:
            details["work_title"] = _find_detail_field_value(work_item_table, "Title")
            details["work_description"] = _find_detail_field_value(work_item_table, "Work Description")
            details["tender_value"] = _find_detail_field_value(work_item_table, "Tender Value in ₹")
            details["product_category"] = _find_detail_field_value(work_item_table, "Product Category")
            details["sub_category"] = _find_detail_field_value(work_item_table, "Sub category")
            details["contract_type"] = _find_detail_field_value(work_item_table, "Contract Type")
            details["bid_validity_days"] = _find_detail_field_value(work_item_table, "Bid Validity(Days)")
            details["period_of_work_days"] = _find_detail_field_value(work_item_table, "Period Of Work(Days)")
            details["location"] = _find_detail_field_value(work_item_table, "Location")
            details["pincode"] = _find_detail_field_value(work_item_table, "Pincode")
            details["pre_bid_meeting_address"] = _find_detail_field_value(work_item_table, "Pre Bid Meeting Address")
            details["bid_opening_place"] = _find_detail_field_value(work_item_table, "Bid Opening Place")
            details["allow_preferential_bidder"] = _find_detail_field_value(work_item_table, "Allow Preferential Bidder")

        # --- Section: Critical Dates ---
        critical_dates_table = find_data_table_after_header(["Critical Dates"])
        if critical_dates_table:
            details["published_date"] = _find_detail_field_value(critical_dates_table, "Published Date")
            details["doc_download_start_date"] = _find_detail_field_value(critical_dates_table, "Document Download / Sale Start Date")
            details["clarification_start_date"] = _find_detail_field_value(critical_dates_table, "Clarification Start Date")
            details["bid_submission_start_date"] = _find_detail_field_value(critical_dates_table, "Bid Submission Start Date")
            details["detail_bid_opening_date"] = _find_detail_field_value(critical_dates_table, "Bid Opening Date")
            details["doc_download_end_date"] = _find_detail_field_value(critical_dates_table, "Document Download / Sale End Date")
            details["clarification_end_date"] = _find_detail_field_value(critical_dates_table, "Clarification End Date")
            details["bid_submission_end_date"] = _find_detail_field_value(critical_dates_table, "Bid Submission End Date")

        # --- Section: Tender Documents ---
        doc_header = soup.find('td', class_='pageheader', string=re.compile(r'Tenders Documents'))
        if doc_header:
            doc_container = doc_header.find_parent('tr').find_next_sibling('tr')
            if doc_container:
                doc_list = []
                doc_tables = doc_container.find_all('table', id=True)
                for table in doc_tables:
                    rows = table.find_all('tr')
                    for i, row in enumerate(rows):
                        if i == 0: continue
                        cols = row.find_all('td')
                        if len(cols) >= 4:
                            link_cell = cols[1]
                            link_tag = link_cell.find('a')
                            if link_tag:
                                doc_list.append({
                                    'name': get_safe_text(link_tag),
                                    'link': urljoin(site_domain, link_tag.get('href', '')),
                                    'description': get_safe_text(cols[2]),
                                    'size': get_safe_text(cols[3]),
                                })
                details["tender_documents"] = doc_list
        
        # --- Section: Tender Inviting Authority ---
        tia_header = soup.find('td', class_='pageheader', string=re.compile(r'Tender Inviting Authority'))
        if tia_header:
            tia_table = tia_header.find_parent('tr').find_next_sibling('tr').find('table', class_='tablebg')
            if tia_table:
                details["tender_inviting_authority_name"] = _find_detail_field_value(tia_table, "Name")
                details["tender_inviting_authority_address"] = _find_detail_field_value(tia_table, "Address")
        
        return details

    except PlaywrightTimeout:
        if scraper_logger: scraper_logger.warning(f"    ⚠️ Timeout scraping detail page: {detail_url}")
    except Exception as e:
        if scraper_logger: scraper_logger.error(f"    ❌ Error scraping detail page {detail_url}: {type(e).__name__} - {e}", exc_info=False)
    return None

async def _scrape_detail_page_with_semaphore(semaphore: asyncio.Semaphore, context: BrowserContext, detail_url: str, site_domain: str, site_key: str, detail_page_timeout_ms: int) -> Optional[Dict[str, Any]]:
    async with semaphore:
        await asyncio.sleep(random.uniform(0.3, 1.0))
        page: Optional[Page] = None
        try:
            page = await context.new_page()
            return await _scrape_detail_page(page, detail_url, site_domain, site_key, detail_page_timeout_ms)
        finally:
            if page and not page.is_closed(): await page.close()

# --- Database Interaction ---
def save_tender_to_db(db: Session, tender_data: Dict[str, Any], site_key: str) -> bool:
    tender_id_str = tender_data.get("detail_tender_id") or tender_data.get("list_page_tender_id")
    if not tender_id_str or tender_id_str == "N/A":
        if scraper_logger: scraper_logger.warning(f"Skipping DB save: No valid Tender ID found for item '{tender_data.get('list_page_title', 'No Title')}'")
        return False
    
    try:
        tender_record = db.query(Tender).filter(Tender.tender_id == tender_id_str).first()
        
        if tender_record:
            db_logger.debug(f"Tender ID '{tender_id_str}' already exists. Updating record.")
            tender_record.tender_title = tender_data.get("work_title") or tender_data.get("list_page_title")
            tender_record.organisation_chain = tender_data.get("organisation_chain") or tender_data.get("list_page_org_chain")
            tender_record.tender_value_numeric = parse_amount_to_numeric(tender_data.get("tender_value"))
            tender_record.emd_amount_numeric = parse_amount_to_numeric(tender_data.get("emd_amount"))
            tender_record.published_date = parse_date_reg(tender_data.get("published_date") or tender_data.get("epublished_date_str"))
            tender_record.closing_date = parse_date_reg(tender_data.get("bid_submission_end_date") or tender_data.get("closing_date_str"))
            tender_record.opening_date = parse_date_reg(tender_data.get("detail_bid_opening_date") or tender_data.get("opening_date_str"))
            tender_record.location = tender_data.get("location")
            tender_record.pincode = tender_data.get("pincode")
            tender_record.full_details_json = tender_data
        else:
            db_logger.info(f"Creating new tender record for ID '{tender_id_str}'.")
            tender_record = Tender(
                tender_id=tender_id_str, source_site=site_key,
                tender_title=tender_data.get("work_title") or tender_data.get("list_page_title"),
                organisation_chain=tender_data.get("organisation_chain") or tender_data.get("list_page_org_chain"),
                tender_value_numeric=parse_amount_to_numeric(tender_data.get("tender_value")),
                emd_amount_numeric=parse_amount_to_numeric(tender_data.get("emd_amount")),
                published_date=parse_date_reg(tender_data.get("published_date") or tender_data.get("epublished_date_str")),
                closing_date=parse_date_reg(tender_data.get("bid_submission_end_date") or tender_data.get("closing_date_str")),
                opening_date=parse_date_reg(tender_data.get("detail_bid_opening_date") or tender_data.get("opening_date_str")),
                location=tender_data.get("location"), pincode=tender_data.get("pincode"),
                status="Live", full_details_json=tender_data
            )
            db.add(tender_record)
        return True
        
    except SQLAlchemyError as e:
        db_logger.error(f"Database error while saving tender ID '{tender_id_str}': {e}", exc_info=False)
        return False
    except Exception as e:
        db_logger.error(f"Unexpected error while saving tender ID '{tender_id_str}' for DB: {e}", exc_info=True)
        return False

# --- Core fetching logic ---

async def fetch_single_list_page(context: BrowserContext, page_number: int, site_base_url: str, site_domain_str: str, site_key_str: str, detail_semaphore: asyncio.Semaphore, page_load_timeout_ms: int, detail_page_timeout_ms: int, retry_limit_val: int) -> Tuple[int, List[Dict[str, Any]]]:
    """Fetches a single list page, scrapes details, and returns a list of combined data dictionaries."""
    list_page_url_to_fetch = site_base_url.format(page_number)
    page_for_list: Optional[Page] = None
    
    await asyncio.sleep(random.uniform(0.5, 2.0))
    for attempt in range(1, retry_limit_val + 2):
        try:
            async with asyncio.timeout(page_load_timeout_ms / 1000 + 15):
                if not page_for_list or page_for_list.is_closed():
                    page_for_list = await context.new_page()

                if scraper_logger: scraper_logger.info(f"📄 Fetching List Page {page_number} (Attempt {attempt}/{retry_limit_val + 1})")
                
                # --- NEW: Pre-navigation and Popup Handling Logic ---
                # On the first page of a batch, visit the homepage first to establish a session.
                if page_number == 1 or attempt > 1:
                    if scraper_logger: scraper_logger.debug(f"  P{page_number}:A{attempt} - Navigating to main site homepage first...")
                    await page_for_list.goto(site_domain_str, wait_until="domcontentloaded", timeout=page_load_timeout_ms)
                    
                    # Handle "Important Message" popup
                    important_message_popup_close_button = page_for_list.locator("div#msgbox button:has-text('Close')")
                    if await important_message_popup_close_button.is_visible(timeout=5000):
                        if scraper_logger: scraper_logger.info("  ℹ️ 'Important Message' popup detected. Clicking close.")
                        await important_message_popup_close_button.click()
                        await page_for_list.wait_for_timeout(1000) # Wait for popup to animate out
                
                # Handle "Session Timed Out" page before proceeding
                if "Your session has timed out" in await page_for_list.content():
                    if scraper_logger: scraper_logger.info("  ℹ️ Session timeout page detected. Clicking restart.")
                    await page_for_list.locator("a:has-text('restart')").click()
                    await page_for_list.wait_for_load_state("domcontentloaded", timeout=page_load_timeout_ms)
                # --- END of Pre-navigation Logic ---

                if scraper_logger: scraper_logger.debug(f"  P{page_number}:A{attempt} - Navigating to target list page URL...")
                await page_for_list.goto(list_page_url_to_fetch, wait_until="domcontentloaded", timeout=page_load_timeout_ms)
                if scraper_logger: scraper_logger.debug(f"  P{page_number}:A{attempt} - Navigation complete. Getting page content...")

                current_page_content = await page_for_list.content()
                if not current_page_content:
                    raise ValueError("Page returned empty content.")

                (PROJECT_ROOT / f"DEBUG__{site_key_str}_Page_{page_number}.html").write_text(current_page_content, encoding='utf-8')
                
                if scraper_logger: scraper_logger.debug(f"  P{page_number}:A{attempt} - Content received and saved to DEBUG file. Parsing with BeautifulSoup...")
                
                soup_list_page = BeautifulSoup(current_page_content, "html.parser")
                
                body_text_lower = soup_list_page.body.get_text().lower() if soup_list_page.body else ""
                if "no records found" in body_text_lower or "no tenders available" in body_text_lower:
                    if scraper_logger: scraper_logger.info(f"  📄 Page {page_number}: 'No records' text found. Stopping fetch.")
                    if page_for_list and not page_for_list.is_closed(): await page_for_list.close()
                    return page_number, [{"status_signal": "NO_RECORDS"}]

                tender_list_rows = soup_list_page.select("table#table tr[id^='informal']")
                if not tender_list_rows:
                    if scraper_logger: scraper_logger.warning(f"  ⚠️ Page {page_number}: Content loaded but no tender rows found. Stopping fetch.")
                    if page_for_list and not page_for_list.is_closed(): await page_for_list.close()
                    return page_number, [{"status_signal": "NO_RECORDS"}]
                
                list_page_tender_items: List[Dict[str, Any]] = []
                for row_item in tender_list_rows:
                    cols_item = row_item.find_all("td")
                    if len(cols_item) < 6: continue
                    title_id_cell_item = cols_item[4]; link_tag_item = title_id_cell_item.find("a", href=True)
                    basic_tender_info = {"list_page_tender_id": "N/A", "list_page_title": "N/A","epublished_date_str": get_safe_text(cols_item[1]), "closing_date_str": get_safe_text(cols_item[2]),"opening_date_str": get_safe_text(cols_item[3]), "list_page_org_chain": get_safe_text(cols_item[5]), "detail_page_link": "N/A"}
                    if link_tag_item:
                        relative_href = link_tag_item.get('href', '')
                        if relative_href and 'javascript:void' not in relative_href: basic_tender_info["detail_page_link"] = urljoin(site_domain_str, relative_href.strip())
                        basic_tender_info["list_page_title"] = clean_text(link_tag_item.get_text())
                    cell_text = title_id_cell_item.get_text(separator='\n', strip=True)
                    all_brackets = BRACKET_CONTENT_REGEX.findall(cell_text)
                    for content in all_brackets:
                        if STRICT_ID_CONTENT_REGEX.match(content.strip()):
                            basic_tender_info["list_page_tender_id"] = content.strip(); break
                    list_page_tender_items.append(basic_tender_info)

                detail_tasks = [_scrape_detail_page_with_semaphore(detail_semaphore, context, item.get("detail_page_link", ""), site_domain_str, site_key_str, detail_page_timeout_ms) for item in list_page_tender_items if item.get("detail_page_link")]
                if scraper_logger: scraper_logger.info(f"  ⏯️ Gathering {len(detail_tasks)} detail scrape tasks for page {page_number}.")
                detail_results = await asyncio.gather(*detail_tasks, return_exceptions=True)

                combined_results = []
                for i, item_data in enumerate(list_page_tender_items):
                    full_data = item_data.copy()
                    if i < len(detail_results):
                        res = detail_results[i]
                        if isinstance(res, dict): full_data.update(res)
                        elif isinstance(res, Exception): 
                            if scraper_logger: scraper_logger.error(f"  ❌ Detail scrape task failed for {item_data.get('list_page_tender_id')}: {res}", exc_info=False)
                    combined_results.append(full_data)

                if scraper_logger: scraper_logger.info(f"  ✅ List Page {page_number} processed successfully ({len(list_page_tender_items)} items).")
                if page_for_list and not page_for_list.is_closed(): await page_for_list.close()
                return page_number, combined_results
        
        except asyncio.TimeoutError:
            if scraper_logger: scraper_logger.error(f"  ❌ MASTER TIMEOUT on List Page {page_number}, attempt {attempt}/{retry_limit_val + 1}. The operation took too long.")
        except (PlaywrightTimeout, ConnectionError) as e:
            if scraper_logger: scraper_logger.warning(f"  ⚠️ Network/Playwright Timeout on List Page {page_number}, attempt {attempt}/{retry_limit_val + 1}: {type(e).__name__}")
        except Exception as e:
            if scraper_logger: scraper_logger.error(f"  ❌ Unhandled Error on List Page {page_number}, attempt {attempt}/{retry_limit_val + 1}: {e}", exc_info=True)
        
        if attempt <= retry_limit_val: await asyncio.sleep(3 * (2 ** (attempt - 1)))
        
    if page_for_list and not page_for_list.is_closed(): await page_for_list.close()
    if scraper_logger: scraper_logger.error(f"  ❌ Failed to process List Page {page_number} after all attempts.")
    return page_number, []

# --- Main orchestration logic for a site ---
async def fetch_all_pages_for_site_and_save_to_db(playwright_instance: Any, site_key: str, site_base_url: str, site_domain: str, max_pages: int, concurrency_val: int, detail_concurrency_val: int, page_timeout: int, detail_timeout: int, retry_max: int):
    browser: Optional[BrowserContext] = None
    context: Optional[BrowserContext] = None
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
            
            unique_tenders_in_batch: Dict[str, Dict[str, Any]] = {}
            for _, tender_list in page_results:
                if not tender_list: 
                    consecutive_empty_pages += 1
                elif tender_list[0].get("status_signal") == "NO_RECORDS":
                    stop_fetching = True; break
                else:
                    consecutive_empty_pages = 0
                    for tender_data in tender_list:
                        primary_id = tender_data.get("detail_tender_id") or tender_data.get("list_page_tender_id")
                        if primary_id and primary_id != "N/A":
                            if primary_id not in unique_tenders_in_batch:
                                unique_tenders_in_batch[primary_id] = tender_data
            
            if stop_fetching or consecutive_empty_pages >= concurrency_val:
                if scraper_logger and consecutive_empty_pages >= concurrency_val: scraper_logger.warning(f"Stopping site {site_key}: {consecutive_empty_pages} consecutive empty/failed pages.")
                break
            
            if unique_tenders_in_batch:
                db: Session = SessionLocal()
                try:
                    for tender_data in unique_tenders_in_batch.values():
                        if save_tender_to_db(db, tender_data, site_key):
                            tenders_saved_count += 1
                        else:
                            tenders_failed_count += 1
                    db.commit()
                    db_logger.info(f"Committed batch of {len(unique_tenders_in_batch)} unique tenders for site '{site_key}'.")
                except Exception as e:
                    db.rollback()
                    db_logger.error(f"Batch DB commit failed for site '{site_key}': {e}", exc_info=True)
                    tenders_failed_count += len(unique_tenders_in_batch)
                finally:
                    db.close()

    except Exception as e:
        if scraper_logger: scraper_logger.critical(f"💥 Unhandled exception during site fetching for {site_key}: {e}", exc_info=True)
    finally:
        if context: await context.close()
        if browser: await browser.close()
        if scraper_logger: scraper_logger.info(f"Scrape run for site {site_key} finished. Saved: {tenders_saved_count}, Failed to save: {tenders_failed_count}.")

# --- Main site entry point ---
async def run_scrape_for_one_site(site_key: str, site_config: Dict[str, str], global_settings: Dict[str, Any]):
    current_site_logger = setup_site_specific_logging(site_key)
    current_site_logger.info(f"🚀 Starting DB-integrated scrape run for site: {site_key}...")
    start_time_site = datetime.datetime.now()
    
    site_base_url = site_config.get("base_url")
    site_domain = site_config.get("domain")
    if not all([site_base_url, site_domain]):
        current_site_logger.error(f"Config for '{site_key}' missing 'base_url' or 'domain'. Skipping.")
        return

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
