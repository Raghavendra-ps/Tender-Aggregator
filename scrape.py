import asyncio
import datetime
import logging
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, Set, List, Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag, NavigableString # NavigableString might not be explicitly used, but good to keep if bs4 needs it
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout, Page, BrowserContext

# === CONFIGURATION (Defaults, overridden by controller via global_settings) ===
MAX_PAGES_TO_FETCH_DEFAULT = 100
RETRY_LIMIT_DEFAULT = 3
CONCURRENCY_DEFAULT = 5
DETAIL_PAGE_CONCURRENCY_LIMIT_DEFAULT = 2
PAGE_LOAD_TIMEOUT_DEFAULT = 75000
DETAIL_PAGE_TIMEOUT_DEFAULT = 75000

# --- NEW DIRECTORY STRUCTURE CONSTANTS ---
try:
    PROJECT_ROOT = Path(__file__).parent.resolve()
except NameError: # pragma: no cover
    PROJECT_ROOT = Path('.').resolve()

SITE_DATA_ROOT = PROJECT_ROOT / "site_data" # Overall data directory

# REG Specific Paths (scrape.py primarily deals with these)
REG_DATA_DIR = SITE_DATA_ROOT / "REG"
REG_RAW_PAGES_DIR_BASE = REG_DATA_DIR / "RawPages" # Base for site-specific raw page subdirs
REG_MERGED_SITE_SPECIFIC_DIR = REG_DATA_DIR / "MergedSiteSpecific" # For Merged_{SITE_KEY}_{DATE}.txt

# LOGS Path (Unified)
LOGS_BASE_DIR = PROJECT_ROOT / "LOGS"
# --- END NEW DIRECTORY STRUCTURE CONSTANTS ---


# === LOG CONFIG (Site-specific logger setup) ===
scraper_logger: Optional[logging.Logger] = None
TODAY_STR = datetime.datetime.now().strftime("%Y-%m-%d") # Using datetime.datetime for consistency if time part was ever needed

def setup_site_specific_logging(site_key: str) -> logging.Logger:
    """Sets up and returns a logger instance specific to the given site_key for scrape.py."""
    global scraper_logger

    # MODIFIED: New log path structure
    regular_scraper_log_dir = LOGS_BASE_DIR / "regular_scraper"
    regular_scraper_log_dir.mkdir(parents=True, exist_ok=True)

    safe_site_key_log = re.sub(r'[^\w\-]+', '_', site_key)
    log_file_path = regular_scraper_log_dir / f"scrape_{safe_site_key_log}.log" # Filename remains scrape_{SITE_KEY}.log

    logger_name = f'scraper_regular_{site_key}' # More specific logger name
    logger = logging.getLogger(logger_name)

    if logger.hasHandlers():
        # logger.handlers.clear() # Optionally clear if re-configuring the same logger instance.
                                # For distinct runs via site_controller, this might not be strictly needed
                                # if site_controller calls this once per site.
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

    header_line = f"\n\n======== {TODAY_STR} / Site: {site_key} (Regular Scrape) ========\n"
    try:
        header_exists = False
        if log_file_path.exists() and log_file_path.stat().st_size > 0: # Check size to avoid error on empty file
             with open(log_file_path, "r", encoding='utf-8', errors='ignore') as f_check: # errors='ignore' for robustness
                 f_check.seek(max(0, log_file_path.stat().st_size - 256)) # Check last ~256 bytes
                 if header_line.strip() in f_check.read():
                     header_exists = True
        if not header_exists:
            with open(log_file_path, "a", encoding='utf-8') as f_append:
                f_append.write(header_line)
    except Exception as e:
        logger.warning(f"Could not ensure date header in log file {log_file_path}: {e}")

    return logger
# === END LOG CONFIG ===

# --- Regex Definitions ---
# (Keep these as they are, no path changes)
BRACKET_CONTENT_REGEX = re.compile(r"\[(.*?)\]")
STRICT_ID_CONTENT_REGEX = re.compile(r"^\d{4}_\w+_\d+_\d+$")

# --- Helper Functions ---
def get_safe_text(element: Optional[Tag], default="N/A", strip=True) -> str:
    """Safely extracts text from a BeautifulSoup Tag."""
    if not element: return default
    try:
        text = element.get_text(strip=strip)
        return text if text else default
    except Exception: return default # Silently return default on error

def clean_text(text: Optional[str]) -> str:
    """Cleans text by replacing multiple whitespaces and stripping."""
    if text is None or text == "N/A": return "N/A"
    # Ensure input is treated as string, replace multiple whitespace with single space
    return re.sub(r'\s+', ' ', str(text)).strip()

def natural_sort_key_for_site_files(filename: Path) -> int:
    """Generates a sort key (page number) for sorting raw page files."""
    match = re.search(r"Page_(\d+)\.txt$", filename.name) # More specific regex
    return int(match.group(1)) if match else 0

def _find_detail_field_value(container: Optional[Tag], *caption_texts: str) -> str:
    """Finds the value of a field in a detail page table based on its caption(s).
       Tries multiple caption texts if provided."""
    if not container: return "N/A"
    for caption_text in caption_texts:
        try:
            # Match caption text more loosely, ignoring content within parentheses and stripping extra whitespace
            # This helps with variations like "Processing Fee in ₹ (18.00% GST Incl.)"
            core_caption = re.sub(r'\s*\(.*?\)\s*', '', caption_text).strip()
            core_caption_pattern = re.compile(r'\s*' + re.escape(core_caption) + r'\s*:?\s*', re.IGNORECASE)
            
            caption_td = container.find(
                lambda tag: tag.name == 'td' and
                            'td_caption' in tag.get('class', []) and
                            core_caption_pattern.search(re.sub(r'\s*\(.*?\)\s*', '', tag.get_text(strip=True)).strip())
            )
            if caption_td:
                next_td = caption_td.find_next_sibling('td', class_='td_field')
                if next_td: return clean_text(next_td.get_text())
                parent_row = caption_td.find_parent('tr')
                if parent_row:
                    field_td = parent_row.find('td', class_='td_field')
                    if field_td and field_td != caption_td:
                        return clean_text(field_td.get_text())
                all_tds_after = caption_td.find_all_next('td')
                for td_field_candidate in all_tds_after:
                    if 'td_field' in td_field_candidate.get('class', []) and \
                       'td_caption' not in td_field_candidate.get('class', []):
                        return clean_text(td_field_candidate.get_text())
            if caption_td: return "N/A" 
        except Exception as e:
            if scraper_logger: scraper_logger.debug(f"  [Detail Scrape] Minor error finding field for caption '{caption_text}': {type(e).__name__}", exc_info=False)
    return "N/A" 

def _format_tag_site(key: str, value: Any, site_key_for_log: str) -> str:
    """Formats key-value pair into XML-like tag(s) for the output file."""
    tag_name = ''.join(word.capitalize() for word in key.split('_'))
    output_lines: List[str] = []

    if isinstance(value, list):
        if key == "tender_documents" and value and all(isinstance(item, dict) for item in value):
            for doc_dict in value:
                line = (f"<DocumentName>{clean_text(doc_dict.get('name', 'N/A'))}</DocumentName>"
                        f"<DocumentLink>{clean_text(doc_dict.get('link', 'N/A'))}</DocumentLink>"
                        f"<DocumentDescription>{clean_text(doc_dict.get('description', 'N/A'))}</DocumentDescription>"
                        f"<DocumentSize>{clean_text(doc_dict.get('size', 'N/A'))}</DocumentSize>")
                if clean_text(doc_dict.get('name', 'N/A')) != "N/A" or clean_text(doc_dict.get('link', 'N/A')) != "N/A":
                    output_lines.append(line)
        elif key == "payment_instruments" and value and all(isinstance(item, dict) for item in value):
            for pi_dict in value:
                sno = clean_text(pi_dict.get('s_no', 'N/A'))
                itype = clean_text(pi_dict.get('instrument_type', 'N/A'))
                if sno != "N/A" and itype != "N/A":
                    output_lines.append(f"<PaymentInstrumentSNo>{sno}</PaymentInstrumentSNo><PaymentInstrumentType>{itype}</PaymentInstrumentType>")
        elif key == "covers_info" and value and all(isinstance(item, dict) for item in value):
            for cover_dict in value:
                 cno = clean_text(cover_dict.get('cover_no', 'N/A'))
                 ctype = clean_text(cover_dict.get('cover_type', 'N/A'))
                 if cno != "N/A" and ctype != "N/A":
                    output_lines.append(f"<CoverInfoNo>{cno}</CoverInfoNo><CoverInfoType>{ctype}</CoverInfoType>")
        else: 
            for item in value:
                cleaned_item = clean_text(str(item))
                if cleaned_item and cleaned_item != 'N/A':
                    output_lines.append(f"<{tag_name}>{cleaned_item}</{tag_name}>")
    elif value is not None and str(value).strip() and str(value) != "N/A":
        cleaned_value = clean_text(str(value))
        if cleaned_value and cleaned_value != 'N/A':
            output_lines.append(f"<{tag_name}>{cleaned_value}</{tag_name}>")

    return "\n".join(output_lines).strip()


# --- Detail Page Scraping ---
async def _scrape_detail_page_with_semaphore(
    semaphore: asyncio.Semaphore,
    context: BrowserContext, 
    detail_url: str,
    site_domain: str, 
    site_key: str, 
    detail_page_timeout_ms: int
) -> Optional[Dict[str, Any]]:
    async with semaphore:
        await asyncio.sleep(0.3 + 0.2 * (hash(detail_url) % 5 / 10.0))
        page: Optional[Page] = None
        try:
            page = await context.new_page()
            result = await _scrape_detail_page(page, detail_url, site_domain, site_key, detail_page_timeout_ms)
            return result
        finally:
            if page and not page.is_closed():
                await page.close()

async def _scrape_detail_page(
    page: Page,
    detail_url: str,
    site_domain: str, 
    site_key: str, 
    detail_page_timeout_ms: int
) -> Optional[Dict[str, Any]]:
    details: Dict[str, Any] = {}
    try:
        if scraper_logger: scraper_logger.info(f"    ➡️  Scraping Detail Page: {detail_url}")
        await page.goto(detail_url, wait_until="domcontentloaded", timeout=detail_page_timeout_ms)
        html = await page.content()
        if not html:
            if scraper_logger: scraper_logger.warning(f"    ⚠️ No content received for detail page: {detail_url}")
            return None

        soup = BeautifulSoup(html, 'html.parser')

        details = {
            "organisation_chain": "N/A", "tender_reference_number": "N/A", "detail_tender_id": "N/A",
            "withdrawal_allowed": "N/A", "tender_type": "N/A", "form_of_contract": "N/A",
            "tender_category": "N/A", "no_of_covers": "N/A", "general_tech_evaluation": "N/A",
            "itemwise_tech_evaluation": "N/A", "payment_mode": "N/A", "multi_currency_boq": "N/A",
            "multi_currency_fee": "N/A", "two_stage_bidding": "N/A",
            "payment_instruments": [], 
            "covers_info": [],         
            "tender_fee": "N/A", 
            "processing_fee": "N/A", # Added
            "fee_payable_to": "N/A", "fee_payable_at": "N/A",
            "tender_fee_exemption": "N/A", "emd_amount": "N/A", "emd_exemption": "N/A",
            "emd_fee_type": "N/A", "emd_percentage": "N/A", "emd_payable_to": "N/A",
            "emd_payable_at": "N/A", "work_title": "N/A", "work_description": "N/A",
            "nda_pre_qualification": "N/A", 
            "independent_external_monitor_remarks": "N/A", # Added
            "tender_value": "N/A", "product_category": "N/A",
            "sub_category": "N/A", "contract_type": "N/A", "bid_validity_days": "N/A",
            "period_of_work_days": "N/A", "location": "N/A", "pincode": "N/A",
            "pre_bid_meeting_place": "N/A", "pre_bid_meeting_address": "N/A", "pre_bid_meeting_date": "N/A",
            "bid_opening_place": "N/A", "allow_nda_tender": "N/A", "allow_preferential_bidder": "N/A",
            "published_date": "N/A", "detail_bid_opening_date": "N/A", "doc_download_start_date": "N/A",
            "doc_download_end_date": "N/A", "clarification_start_date": "N/A", "clarification_end_date": "N/A",
            "bid_submission_start_date": "N/A", "bid_submission_end_date": "N/A",
            "tender_documents": [], 
            "tender_inviting_authority_name": "N/A", "tender_inviting_authority_address": "N/A"
        }

        def find_data_table_after_header(header_text_variants: List[str]) -> Optional[Tag]:
             table: Optional[Tag] = None
             for header_text in header_text_variants:
                 # Find the <td> that contains the header text and has class 'pageheader'
                 header_td_element = soup.find(
                     lambda tag: tag.name == "td" and
                                 header_text.lower() in tag.get_text(strip=True).lower() and
                                 "pageheader" in tag.get('class', [])
                 )
                 if header_td_element:
                      # The actual data table is usually the next sibling <table> of the <table> containing the header row.
                      # Or sometimes it's inside the next <tr>'s <td>
                      # Let's find the parent <tr> of the header <td>
                      header_row = header_td_element.find_parent('tr')
                      if header_row:
                          # Case 1: Data table is in the next row, within a <td>, then a <table>
                          next_row = header_row.find_next_sibling('tr')
                          if next_row:
                              data_cell_in_next_row = next_row.find('td')
                              if data_cell_in_next_row:
                                  table = data_cell_in_next_row.find('table', class_=re.compile(r'(tablebg|list_table)'))
                                  if table: return table
                          
                          # Case 2: Data table is a sibling of the header's parent table (less common for these section headers)
                          # This logic might need to be more specific if Case 1 fails consistently
                          # For now, we rely on the structure seen in Page.html where data table is in next <tr><td><table>
                      if table: return table 
             if not table and scraper_logger:
                  scraper_logger.debug(f"    [Detail Page - {site_key}] Table not found after headers '{', '.join(header_text_variants)}' for {detail_url}")
             return None

        basic_details_table = find_data_table_after_header(["Basic Details"])
        if basic_details_table:
            details["organisation_chain"] = _find_detail_field_value(basic_details_table, "Organisation Chain")
            details["tender_reference_number"] = _find_detail_field_value(basic_details_table, "Tender Reference Number")
            details["detail_tender_id"] = _find_detail_field_value(basic_details_table, "Tender ID")
            details["withdrawal_allowed"] = _find_detail_field_value(basic_details_table, "Withdrawal Allowed")
            details["tender_type"] = _find_detail_field_value(basic_details_table, "Tender Type")
            details["form_of_contract"] = _find_detail_field_value(basic_details_table, "Form Of Contract")
            details["tender_category"] = _find_detail_field_value(basic_details_table, "Tender Category")
            details["no_of_covers"] = _find_detail_field_value(basic_details_table, "No. of Covers")
            details["general_tech_evaluation"] = _find_detail_field_value(basic_details_table, "General Technical Evaluation Allowed")
            details["itemwise_tech_evaluation"] = _find_detail_field_value(basic_details_table, "ItemWise Technical Evaluation Allowed")
            details["payment_mode"] = _find_detail_field_value(basic_details_table, "Payment Mode")
            details["multi_currency_boq"] = _find_detail_field_value(basic_details_table, "Is Multi Currency Allowed For BOQ")
            details["multi_currency_fee"] = _find_detail_field_value(basic_details_table, "Is Multi Currency Allowed For Fee")
            details["two_stage_bidding"] = _find_detail_field_value(basic_details_table, "Allow Two Stage Bidding")

        payment_instruments_table = find_data_table_after_header(["Payment Instruments"])
        if payment_instruments_table:
            pi_list = []
            # The table structure is: <table><tbody><tr><td><table id="onlineInstrumentsTableView">...</table></td></tr></tbody></table>
            # So we need to find the inner table.
            inner_pi_table = payment_instruments_table.find('table', id='onlineInstrumentsTableView')
            if inner_pi_table:
                rows = inner_pi_table.find_all('tr') # Get all rows, including header
                for row_idx, row in enumerate(rows):
                    if row_idx == 0 and row.find('td', class_='list_header'): continue # Skip header if class list_header
                    if row_idx == 0 and row.find('th'): continue # Skip header if <th>
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        s_no = clean_text(cols[0].get_text())
                        instrument_type = clean_text(cols[1].get_text())
                        if s_no and s_no != "N/A" and instrument_type and instrument_type != "N/A":
                             pi_list.append({"s_no": s_no, "instrument_type": instrument_type})
            details["payment_instruments"] = pi_list

        covers_table = find_data_table_after_header(["Covers Information", "Cover Details"])
        if covers_table:
            ci_list = []
            inner_covers_table = covers_table.find('table', id='packetTableView') # As per Page.html
            if inner_covers_table:
                rows = inner_covers_table.find_all('tr')
                for row_idx, row in enumerate(rows):
                    if row_idx == 0 and row.find('td', class_='list_header'): continue 
                    if row_idx == 0 and row.find('th'): continue 
                    cols = row.find_all('td')
                    if len(cols) >= 2: # Cover No, Cover Type
                        cover_no_val = clean_text(cols[0].get_text()) 
                        cover_type = clean_text(cols[1].get_text())
                        if cover_no_val and cover_no_val != "N/A" and cover_type and cover_type != "N/A":
                            ci_list.append({"cover_no": cover_no_val, "cover_type": cover_type})
            details["covers_info"] = ci_list
        
        # For Fee Details, the structure is a bit different in Page.html
        # It has two main <td> elements, one for Tender Fee and one for EMD, each containing a table.
        # We need to find the 'pageheader' for "Tender Fee Details" and "EMD Fee Details" and then parse their respective tables.

        # Tender Fee Details
        tender_fee_header = soup.find(lambda tag: tag.name == "td" and "Tender Fee Details".lower() in tag.get_text(strip=True).lower() and "pageheader" in tag.get('class', []))
        if tender_fee_header:
            tender_fee_table_container = tender_fee_header.find_parent('tr')
            if tender_fee_table_container:
                tender_fee_data_row = tender_fee_table_container.find_next_sibling('tr')
                if tender_fee_data_row:
                    actual_tender_fee_table = tender_fee_data_row.find('table', class_='tablebg')
                    if actual_tender_fee_table:
                        details["tender_fee"] = _find_detail_field_value(actual_tender_fee_table, "Tender Fee in ₹", "Tender Fee")
                        details["processing_fee"] = _find_detail_field_value(actual_tender_fee_table, "Processing Fee in ₹ (18.00% GST Incl.)", "Processing Fee in ₹", "Processing Fee")
                        details["fee_payable_to"] = _find_detail_field_value(actual_tender_fee_table, "Fee Payable To")
                        details["fee_payable_at"] = _find_detail_field_value(actual_tender_fee_table, "Fee Payable At")
                        details["tender_fee_exemption"] = _find_detail_field_value(actual_tender_fee_table, "Tender Fee Exemption Allowed")

        # EMD Fee Details
        emd_fee_header = soup.find(lambda tag: tag.name == "td" and "EMD Fee Details".lower() in tag.get_text(strip=True).lower() and "pageheader" in tag.get('class', []))
        if emd_fee_header:
            emd_fee_table_container = emd_fee_header.find_parent('tr')
            if emd_fee_table_container:
                emd_fee_data_row = emd_fee_table_container.find_next_sibling('tr')
                if emd_fee_data_row:
                    actual_emd_fee_table = emd_fee_data_row.find('table', class_='tablebg')
                    if actual_emd_fee_table:
                        details["emd_amount"] = _find_detail_field_value(actual_emd_fee_table, "EMD Amount in ₹", "EMD Amount")
                        details["emd_exemption"] = _find_detail_field_value(actual_emd_fee_table, "EMD Exemption Allowed")
                        details["emd_fee_type"] = _find_detail_field_value(actual_emd_fee_table, "EMD Fee Type")
                        details["emd_percentage"] = _find_detail_field_value(actual_emd_fee_table, "EMD Percentage")
                        details["emd_payable_to"] = _find_detail_field_value(actual_emd_fee_table, "EMD Payable To")
                        details["emd_payable_at"] = _find_detail_field_value(actual_emd_fee_table, "EMD Payable At")


        work_item_table = find_data_table_after_header(["Work Item Details", "Work/Item(s)"])
        if work_item_table: # This is the main tablebg for work item details
            details["work_title"] = _find_detail_field_value(work_item_table, "Title", "Work Title")
            details["work_description"] = _find_detail_field_value(work_item_table, "Work Description", "Description")
            details["nda_pre_qualification"] = _find_detail_field_value(work_item_table, "NDA/Pre Qualification", "Pre Qualification Details", "Pre Qualification")
            details["independent_external_monitor_remarks"] = _find_detail_field_value(work_item_table, "Independent External Monitor/Remarks")
            details["tender_value"] = _find_detail_field_value(work_item_table, "Tender Value in ₹", "Tender Value")
            details["product_category"] = _find_detail_field_value(work_item_table, "Product Category")
            details["sub_category"] = _find_detail_field_value(work_item_table, "Sub category", "Sub-Category")
            details["contract_type"] = _find_detail_field_value(work_item_table, "Contract Type")
            details["bid_validity_days"] = _find_detail_field_value(work_item_table, "Bid Validity(Days)")
            details["period_of_work_days"] = _find_detail_field_value(work_item_table, "Period Of Work(Days)")
            details["location"] = _find_detail_field_value(work_item_table, "Location")
            details["pincode"] = _find_detail_field_value(work_item_table, "Pincode")
            details["pre_bid_meeting_place"] = _find_detail_field_value(work_item_table, "Pre Bid Meeting Place")
            details["pre_bid_meeting_address"] = _find_detail_field_value(work_item_table, "Pre Bid Meeting Address")
            details["pre_bid_meeting_date"] = _find_detail_field_value(work_item_table, "Pre Bid Meeting Date")
            details["bid_opening_place"] = _find_detail_field_value(work_item_table, "Bid Opening Place")
            details["allow_nda_tender"] = _find_detail_field_value(work_item_table, "Should Allow NDA Tender")
            details["allow_preferential_bidder"] = _find_detail_field_value(work_item_table, "Allow Preferential Bidder")

        critical_dates_table = find_data_table_after_header(["Critical Dates"])
        if critical_dates_table:
            details["published_date"] = _find_detail_field_value(critical_dates_table, "Published Date", "Publish Date")
            details["doc_download_start_date"] = _find_detail_field_value(critical_dates_table, "Document Download / Sale Start Date")
            details["doc_download_end_date"] = _find_detail_field_value(critical_dates_table, "Document Download / Sale End Date")
            details["clarification_start_date"] = _find_detail_field_value(critical_dates_table, "Clarification Start Date")
            details["clarification_end_date"] = _find_detail_field_value(critical_dates_table, "Clarification End Date")
            details["bid_submission_start_date"] = _find_detail_field_value(critical_dates_table, "Bid Submission Start Date")
            details["bid_submission_end_date"] = _find_detail_field_value(critical_dates_table, "Bid Submission End Date")
            details["detail_bid_opening_date"] = _find_detail_field_value(critical_dates_table, "Bid Opening Date")
        
        docs_container_header = soup.find(lambda tag: tag.name == "td" and "Tenders Documents".lower() in tag.get_text(strip=True).lower() and "pageheader" in tag.get('class', []))
        if docs_container_header:
            docs_table_container_row = docs_container_header.find_parent('tr')
            if docs_table_container_row:
                docs_data_row = docs_table_container_row.find_next_sibling('tr')
                if docs_data_row:
                    # This td might contain multiple tables (NIT, WorkItem)
                    main_doc_data_td = docs_data_row.find('td')
                    if main_doc_data_td:
                        tender_documents_list: List[Dict[str,str]] = []
                        
                        # NIT Documents (table id="table")
                        nit_doc_table_header_td = main_doc_data_td.find(lambda tag: tag.name == 'td' and tag.get_text(strip=True) == "NIT Document")
                        if nit_doc_table_header_td:
                            nit_doc_table = nit_doc_table_header_td.find_next('table', id='table')
                            if nit_doc_table:
                                rows = nit_doc_table.find_all('tr')
                                for row_idx, row in enumerate(rows):
                                    if row_idx == 0 and (row.find('th') or 'list_header' in row.get('class', [])): continue
                                    cols = row.find_all('td')
                                    if len(cols) >= 4: # S.No, Name, Desc, Size
                                        doc_name, doc_link, doc_desc, doc_size = "N/A", "N/A", "N/A", "N/A"
                                        link_tag = cols[1].find('a', href=True)
                                        if link_tag:
                                            doc_name = clean_text(link_tag.get_text())
                                            relative_href = link_tag.get('href', '')
                                            doc_link = urljoin(site_domain, relative_href.strip()) if relative_href and relative_href.strip() != '#' else "N/A"
                                        else:
                                            doc_name = clean_text(cols[1].get_text())
                                        doc_desc = clean_text(cols[2].get_text())
                                        doc_size = clean_text(cols[3].get_text())
                                        if doc_name != "N/A" or doc_link != "N/A":
                                            tender_documents_list.append({"name": doc_name, "link": doc_link, "description": doc_desc, "size": doc_size})
                        
                        # Work Item Documents (table id="workItemDocumenttable")
                        work_item_doc_table_header_td = main_doc_data_td.find(lambda tag: tag.name == 'td' and tag.get_text(strip=True) == "Work Item Documents")
                        if work_item_doc_table_header_td:
                            work_item_doc_table = work_item_doc_table_header_td.find_next('table', id='workItemDocumenttable')
                            if work_item_doc_table:
                                rows = work_item_doc_table.find_all('tr')
                                for row_idx, row in enumerate(rows):
                                    if row_idx == 0 and (row.find('th') or 'list_header' in row.get('class', [])): continue
                                    cols = row.find_all('td')
                                    if len(cols) >= 5: # S.No, Doc Type, Name, Desc, Size
                                        doc_name, doc_link, doc_desc, doc_size = "N/A", "N/A", "N/A", "N/A"
                                        # doc_type = clean_text(cols[1].get_text()) # Can be stored if needed
                                        link_tag = cols[2].find('a', href=True)
                                        if link_tag:
                                            doc_name = clean_text(link_tag.get_text())
                                            relative_href = link_tag.get('href', '')
                                            doc_link = urljoin(site_domain, relative_href.strip()) if relative_href and relative_href.strip() != '#' else "N/A"
                                        else:
                                            doc_name = clean_text(cols[2].get_text())
                                        doc_desc = clean_text(cols[3].get_text())
                                        doc_size = clean_text(cols[4].get_text())
                                        if doc_name != "N/A" or doc_link != "N/A":
                                            tender_documents_list.append({"name": doc_name, "link": doc_link, "description": doc_desc, "size": doc_size})
                        details["tender_documents"] = tender_documents_list

        tia_table = find_data_table_after_header(["Tender Inviting Authority", "Inviting Authority"])
        if tia_table:
            details["tender_inviting_authority_name"] = _find_detail_field_value(tia_table, "Name")
            details["tender_inviting_authority_address"] = _find_detail_field_value(tia_table, "Address")
        
        return details

    except PlaywrightTimeout:
        if scraper_logger: scraper_logger.warning(f"    ⚠️ Timeout scraping detail page: {detail_url}")
    except Exception as e:
        if scraper_logger:
            if "net::ERR" in str(e) or "Timeout" in str(e):
                 scraper_logger.info(f"    ℹ️ Network/Timeout error scraping detail page {detail_url}: {type(e).__name__}")
            else: 
                 scraper_logger.error(f"    ❌ Error scraping detail page {detail_url}: {type(e).__name__} - {e}", exc_info=False) 
    return None


async def fetch_single_list_page(
    context: BrowserContext,
    page_number: int,
    site_base_url: str,
    site_domain_str: str,
    site_key_str: str,
    detail_semaphore: asyncio.Semaphore,
    page_load_timeout_ms: int,
    detail_page_timeout_ms: int,
    retry_limit_val: int
) -> Tuple[int, Optional[str]]:
    list_page_url_to_fetch = site_base_url.format(page_number)
    main_site_page_url = urljoin(site_domain_str, urlparse(site_domain_str).path)

    final_tagged_content_for_page = ""
    page_for_list: Optional[Page] = None 

    try:
        page_for_list = await context.new_page() 

        for attempt in range(1, retry_limit_val + 2): 
            list_page_tender_items: List[Dict[str, str]] = []
            try:
                if scraper_logger: scraper_logger.info(f"📄 Fetching List Page {page_number} (Attempt {attempt}/{retry_limit_val + 1})")

                if attempt == 1:
                    try:
                        if scraper_logger: scraper_logger.debug(f"  Pre-visiting main site: {main_site_page_url}")
                        await page_for_list.goto(main_site_page_url, wait_until="load", timeout=page_load_timeout_ms // 2) 
                        await page_for_list.wait_for_timeout(500) 
                    except Exception as e_pre_visit:
                        if scraper_logger: scraper_logger.warning(f"  ⚠️ Failed initial navigation to {main_site_page_url}: {e_pre_visit}. Proceeding...")

                await page_for_list.goto(list_page_url_to_fetch, wait_until="networkidle", timeout=page_load_timeout_ms)
                current_page_content = await page_for_list.content()

                if not current_page_content:
                     if scraper_logger: scraper_logger.warning(f"  ⚠️ No content received for list page {page_number}. Retrying.")
                     await asyncio.sleep(3 * (2 ** (attempt-1))); 
                     continue 

                soup_list_page = BeautifulSoup(current_page_content, "html.parser")
                tender_list_table = soup_list_page.find("table", id="table")

                if not tender_list_table:
                    body_text_lower_list = soup_list_page.body.get_text(separator=' ', strip=True).lower() if soup_list_page.body else ""
                    if "no records found" in body_text_lower_list or "no tenders available" in body_text_lower_list:
                        if scraper_logger: scraper_logger.info(f"  📄 Page {page_number}: No records found on page (body text). Stopping fetch for this site.")
                        return page_number, "--- NO RECORDS ---" 
                    if scraper_logger: scraper_logger.warning(f"  ⚠️ Could not find table with id='table' on list page {page_number}. Retrying.")
                    if attempt <= retry_limit_val:
                        await asyncio.sleep(3 * (2**(attempt-1))); continue
                    else: 
                        if scraper_logger: scraper_logger.error(f"  ❌ No table found on list page {page_number} after all retries.")
                        return page_number, "" 

                tender_list_rows = tender_list_table.find_all("tr", id=re.compile(r"informal"))
                if scraper_logger: scraper_logger.debug(f"  Found {len(tender_list_rows)} potential tender rows on list page {page_number}.")

                if not tender_list_rows:
                     table_text_lower = tender_list_table.get_text(separator=' ', strip=True).lower()
                     if "no records found" in table_text_lower or "no tenders available" in table_text_lower:
                         if scraper_logger: scraper_logger.info(f"  📄 Page {page_number}: No records found (within table text). Stopping fetch for this site.")
                         return page_number, "--- NO RECORDS ---"
                     else:
                         if scraper_logger: scraper_logger.warning(f"  ⚠️ Table found on page {page_number}, but no rows with id 'informal*'. Skipping page.")
                         return page_number, "" 

                row_idx_count = 0
                for row_item in tender_list_rows:
                    row_idx_count += 1; cols_item = row_item.find_all("td")
                    if len(cols_item) < 6: continue 
                    basic_tender_info = {"list_page_tender_id": "N/A", "list_page_title": "N/A","epublished_date_str": get_safe_text(cols_item[1]), "closing_date_str": get_safe_text(cols_item[2]),"opening_date_str": get_safe_text(cols_item[3]), "list_page_org_chain": get_safe_text(cols_item[5]),"detail_page_link": "N/A"}
                    title_id_cell_item = cols_item[4]; cell_text_item = title_id_cell_item.get_text(separator='\n', strip=True); link_tag_item = title_id_cell_item.find("a", href=True)
                    if link_tag_item: 
                        relative_href_item = link_tag_item.get('href', '');
                        if relative_href_item and 'javascript:void' not in relative_href_item and relative_href_item.strip() != '#': basic_tender_info["detail_page_link"] = urljoin(site_domain_str, relative_href_item.strip())
                        basic_tender_info["list_page_title"] = clean_text(link_tag_item.get_text())
                    all_brackets_item = BRACKET_CONTENT_REGEX.findall(cell_text_item); found_id_in_bracket = ""
                    for content_in_bracket in all_brackets_item: 
                        content_in_bracket_stripped = content_in_bracket.strip();
                        if STRICT_ID_CONTENT_REGEX.match(content_in_bracket_stripped):
                            basic_tender_info["list_page_tender_id"] = content_in_bracket_stripped; found_id_in_bracket = f"[{content_in_bracket_stripped}]"
                            if basic_tender_info["list_page_title"] == found_id_in_bracket: basic_tender_info["list_page_title"] = "N/A" 
                            break
                    if basic_tender_info["list_page_title"] == "N/A":
                        title_parts_from_cell = [p.strip() for p in cell_text_item.split('\n') if p.strip()];
                        if title_parts_from_cell:
                             if title_parts_from_cell[0] != found_id_in_bracket: basic_tender_info["list_page_title"] = title_parts_from_cell[0]
                             elif len(title_parts_from_cell) > 1: basic_tender_info["list_page_title"] = title_parts_from_cell[1] 
                    if basic_tender_info["list_page_tender_id"] != "N/A": list_page_tender_items.append(basic_tender_info)
                    else:
                        if scraper_logger: scraper_logger.warning(f"  ⚠️ Skipping row {row_idx_count} on page {page_number} - No Tender ID found. Link: {basic_tender_info['detail_page_link']}")
                
                detail_scraping_tasks = []
                items_for_detail_processing = [] 
                for item_info_list in list_page_tender_items:
                    items_for_detail_processing.append(item_info_list) 
                    detail_link = item_info_list.get("detail_page_link")
                    if detail_link and detail_link != "N/A":
                        detail_scraping_tasks.append(
                            _scrape_detail_page_with_semaphore(
                                detail_semaphore, context, detail_link,
                                site_domain_str, site_key_str, detail_page_timeout_ms
                            )
                        )
                    else:
                         detail_scraping_tasks.append(asyncio.sleep(0, result=None))

                detail_scrape_results: List[Optional[Dict[str, Any]]] = []
                if detail_scraping_tasks:
                    if scraper_logger: scraper_logger.info(f"  ⏯️ Gathering {len(detail_scraping_tasks)} detail scrape tasks for page {page_number} (Semaphore limit: {detail_semaphore._value})...")
                    results_from_gather_details = await asyncio.gather(*detail_scraping_tasks, return_exceptions=True)
                    if scraper_logger: scraper_logger.info(f"  ✅ Finished gathering detail scrape results for page {page_number}.")
                    detail_scrape_results = [] 
                    for i, res_detail in enumerate(results_from_gather_details):
                         item_id_for_log = items_for_detail_processing[i].get('list_page_tender_id','Unknown ID')
                         if isinstance(res_detail, dict): detail_scrape_results.append(res_detail)
                         elif isinstance(res_detail, Exception):
                              if scraper_logger: scraper_logger.error(f"  ❌ Detail scrape task failed for {item_id_for_log} (Page {page_number}): {res_detail}", exc_info=False)
                              detail_scrape_results.append(None) 
                         else: 
                              detail_scrape_results.append(res_detail)

                page_content_blocks_tagged: List[str] = []
                for i, original_item_info in enumerate(items_for_detail_processing):
                    details_scraped = detail_scrape_results[i] if i < len(detail_scrape_results) else None
                    combined_tender_data = original_item_info.copy() 
                    if details_scraped: 
                        combined_tender_data.update(details_scraped)
                    combined_tender_data['source_site_key'] = site_key_str
                    tagged_block_parts_list = ["--- TENDER START ---"]
                    for key_data, value_data in combined_tender_data.items():
                         formatted_tag_str = _format_tag_site(key_data, value_data, site_key_str)
                         if formatted_tag_str: tagged_block_parts_list.append(formatted_tag_str)
                    tagged_block_parts_list.append("--- TENDER END ---")
                    page_content_blocks_tagged.append("\n".join(tagged_block_parts_list))

                final_tagged_content_for_page = "\n\n".join(page_content_blocks_tagged).strip()

                if not final_tagged_content_for_page:
                     if scraper_logger: scraper_logger.info(f"  ⚠️ Page {page_number}: Processed {row_idx_count} rows, but no valid tagged content generated.")
                     return page_number, "" 

                if scraper_logger: scraper_logger.info(f"  ✅ List Page {page_number} processed successfully ({len(list_page_tender_items)} valid items found on list).")
                return page_number, final_tagged_content_for_page

            except PlaywrightTimeout as e_timeout:
                if scraper_logger: scraper_logger.warning(f"  ⚠️ Timeout on List Page {page_number}, attempt {attempt}/{retry_limit_val + 1}: {e_timeout}")
            except ConnectionError as e_conn: 
                if scraper_logger: scraper_logger.warning(f"  ⚠️ Connection Error on List Page {page_number}, attempt {attempt}/{retry_limit_val + 1}: {e_conn}")
            except Exception as e_general:
                if scraper_logger: scraper_logger.error(f"  ❌ Error fetching/processing List Page {page_number}, attempt {attempt}/{retry_limit_val + 1}: {type(e_general).__name__} - {e_general}", exc_info=True)

            if attempt <= retry_limit_val:
                wait_time_retry = 3 * (2 ** (attempt -1)) 
                if scraper_logger: scraper_logger.info(f"  Retrying list page {page_number} in {wait_time_retry} seconds...")
                await asyncio.sleep(wait_time_retry)

        if scraper_logger: scraper_logger.error(f"  ❌ Failed to process List Page {page_number} after {retry_limit_val + 1} attempts.")
        return page_number, None 
    finally:
        if page_for_list and not page_for_list.is_closed():
            await page_for_list.close()

async def fetch_all_pages_for_site(
    playwright_instance: Any,
    site_key: str,
    site_base_url: str,
    site_domain: str,
    max_pages: int,
    concurrency_val: int,
    detail_concurrency_val: int,
    page_timeout: int,
    detail_timeout: int,
    retry_max: int,
    raw_pages_dir_for_site: Path 
):
    browser: Optional[BrowserContext.browser] = None # type: ignore
    context: Optional[BrowserContext] = None 
    all_page_statuses: Dict[int, str] = {}
    intermediate_files_written = 0
    
    try:
        browser = await playwright_instance.chromium.launch(headless=True)
        context = await browser.new_context(ignore_https_errors=True)
        
        if scraper_logger: scraper_logger.info(f"Launched browser context. Max list page concurrency: {concurrency_val}")

        detail_page_semaphore = asyncio.Semaphore(detail_concurrency_val)
        last_valid_page_content_hash: Optional[int] = None
        stop_site_fetching = False
        current_list_page_num = 1
        consecutive_empty_or_failed_list_pages = 0
        consecutive_duplicate_list_pages = 0

        while current_list_page_num <= max_pages and not stop_site_fetching:
            list_page_tasks = []
            batch_start_num = current_list_page_num
            batch_end_num = min(batch_start_num + concurrency_val - 1, max_pages)
            pages_in_this_batch = list(range(batch_start_num, batch_end_num + 1))

            if not pages_in_this_batch: break

            if scraper_logger: scraper_logger.info(f"🚀 Preparing batch for List Pages: {pages_in_this_batch[0]} to {pages_in_this_batch[-1]}")

            for list_page_num_to_fetch in pages_in_this_batch:
                list_page_tasks.append(
                    fetch_single_list_page( 
                        context, list_page_num_to_fetch, site_base_url, site_domain, site_key,
                        detail_page_semaphore, page_timeout, detail_timeout, retry_max
                    )
                )

            current_list_page_num += len(pages_in_this_batch) 

            if not list_page_tasks: break 

            batch_results_list = await asyncio.gather(*list_page_tasks)

            for i, result_tuple in enumerate(batch_results_list):
                actual_page_num_processed = pages_in_this_batch[i]
                
                if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                    if scraper_logger: scraper_logger.error(f"  Unexpected result type for page {actual_page_num_processed}: {type(result_tuple)}. Treating as failure.")
                    tagged_page_content = None
                else:
                     _, tagged_page_content = result_tuple 

                if tagged_page_content is None: 
                    all_page_statuses[actual_page_num_processed] = "Failed"
                    consecutive_empty_or_failed_list_pages += 1
                    if consecutive_empty_or_failed_list_pages >= concurrency_val:
                        stop_site_fetching = True
                        if scraper_logger: scraper_logger.error(f"🛑 Stopping site: {concurrency_val} consecutive list page fetch failures, ending at page {actual_page_num_processed}.")
                        break 
                    continue 

                if tagged_page_content == "--- NO RECORDS ---": 
                    all_page_statuses[actual_page_num_processed] = "No Records Found"
                    stop_site_fetching = True
                    if scraper_logger: scraper_logger.info(f"🏁 Stopping site: 'No Records' signal received on list page {actual_page_num_processed}.")
                    break 

                consecutive_empty_or_failed_list_pages = 0

                if not tagged_page_content.strip(): 
                    all_page_statuses[actual_page_num_processed] = "Processed Empty"
                    last_valid_page_content_hash = None 
                    consecutive_duplicate_list_pages = 0
                    continue 

                current_page_content_hash = hash(tagged_page_content)
                if last_valid_page_content_hash is not None and current_page_content_hash == last_valid_page_content_hash:
                    all_page_statuses[actual_page_num_processed] = "Duplicate"
                    consecutive_duplicate_list_pages += 1
                    if consecutive_duplicate_list_pages >= 2: 
                        stop_site_fetching = True
                        if scraper_logger: scraper_logger.warning(f"🛑 Stopping site: {consecutive_duplicate_list_pages + 1} consecutive identical list pages detected, ending at page {actual_page_num_processed}.")
                        break 
                else:
                    consecutive_duplicate_list_pages = 0 

                save_path_for_page = raw_pages_dir_for_site / f"Page_{actual_page_num_processed}.txt"
                try:
                    save_path_for_page.write_text(tagged_page_content, encoding="utf-8")
                    all_page_statuses[actual_page_num_processed] = "Saved"
                    intermediate_files_written += 1
                    last_valid_page_content_hash = current_page_content_hash
                except OSError as e_save: # pragma: no cover
                    if scraper_logger: scraper_logger.error(f"  ❌ Failed to save intermediate file {save_path_for_page}: {e_save}")
                    all_page_statuses[actual_page_num_processed] = "Save Failed"
                    
            if stop_site_fetching:
                break 

        if current_list_page_num > max_pages:
            if scraper_logger: scraper_logger.info(f"Reached maximum page limit for site: {max_pages}.")
        
    except Exception as e_site_fetch: # pragma: no cover
        if scraper_logger: scraper_logger.critical(f"💥 Unhandled exception during site fetching for {site_key}: {e_site_fetch}", exc_info=True)
    finally:
        if scraper_logger: scraper_logger.info("Attempting to close browser context and browser...")
        if context: await context.close()
        if browser: await browser.close()
        if scraper_logger: scraper_logger.info("Browser closed.")

    if scraper_logger: scraper_logger.info(f"Fetching complete for site {site_key}. Processed up to list page: {current_list_page_num-1}. Saved {intermediate_files_written} intermediate files.")
    return all_page_statuses

async def merge_site_specific_raw_files(
    site_key: str,
    raw_pages_dir_for_site: Path, # This path is like site_data/REG/RawPages/{SITE_KEY}
    # final_output_dir_root argument is now REG_DATA_DIR (site_data/REG/)
    # The function will create "MergedSiteSpecific" inside this.
    final_output_dir_root: Path
) -> Tuple[int, Optional[Path]]:
    today_filename_part = datetime.datetime.now().strftime("%Y-%m-%d")
    safe_site_key_for_file = re.sub(r'[^\w\-]+', '_', site_key)

    # MODIFIED: Use the globally defined constant for clarity, though the logic was already correct
    # Old: site_merged_output_dir = final_output_dir_root / "SiteSpecificMerged"
    # New: Directly use the constant that should resolve to the same path
    site_merged_output_dir = REG_MERGED_SITE_SPECIFIC_DIR
    # Ensure it matches the expectation based on how final_output_dir_root is passed
    if site_merged_output_dir != (final_output_dir_root / "MergedSiteSpecific"):
        if scraper_logger: # Check if logger is initialized
            scraper_logger.warning(f"Path mismatch for site_merged_output_dir. Expected based on constant: {REG_MERGED_SITE_SPECIFIC_DIR}, Derived from arg: {final_output_dir_root / 'SiteSpecificMerged'}")
        # Fallback to using the argument-derived path if there's a mismatch, but log it.
        # This situation implies a configuration inconsistency between constants.
        site_merged_output_dir = final_output_dir_root / "MergedSiteSpecific"


    site_merged_output_dir.mkdir(parents=True, exist_ok=True)
    final_site_output_path = site_merged_output_dir / f"Merged_{safe_site_key_for_file}_{today_filename_part}.txt"

    merged_tender_blocks_for_site = 0
    seen_tender_block_hashes_for_site: Set[int] = set()

    if scraper_logger: scraper_logger.info(f"Merging raw files for site '{site_key}' into: {final_site_output_path}, from: {raw_pages_dir_for_site}")

    if not raw_pages_dir_for_site.is_dir():
        if scraper_logger: scraper_logger.warning(f"Raw pages directory for site '{site_key}' not found: {raw_pages_dir_for_site}")
        return 0, None
    try:
        site_page_files = sorted(raw_pages_dir_for_site.glob("Page_*.txt"), key=natural_sort_key_for_site_files)
        site_page_files_count = len(site_page_files)
        if scraper_logger: scraper_logger.info(f"Found {site_page_files_count} raw page files for site '{site_key}'.")
    except OSError as e_list_site_files:
        if scraper_logger: scraper_logger.error(f"Error listing raw page files for site '{site_key}' in {raw_pages_dir_for_site}: {e_list_site_files}")
        return 0, None

    if not site_page_files:
        if scraper_logger: scraper_logger.warning(f"No raw page files found for site '{site_key}' to merge.")
        try:
            # Only remove if it's truly empty
            if not any(raw_pages_dir_for_site.iterdir()):
                 raw_pages_dir_for_site.rmdir()
                 if scraper_logger: scraper_logger.info(f"Removed empty raw directory for site '{site_key}': {raw_pages_dir_for_site}")
            else:
                 if scraper_logger: scraper_logger.info(f"Raw directory for site '{site_key}' not empty, not removed: {raw_pages_dir_for_site}")
        except OSError: # pass if rmdir fails (e.g. hidden files)
            if scraper_logger: scraper_logger.warning(f"Could not remove raw directory {raw_pages_dir_for_site}, it might not be empty or permission issue.")
        return 0, None

    try:
        with open(final_site_output_path, "w", encoding="utf-8") as outfile_site:
            for site_page_path in site_page_files:
                try:
                    page_file_content = site_page_path.read_text(encoding="utf-8", errors='replace').strip()
                    if not page_file_content:
                        if scraper_logger: scraper_logger.debug(f"Skipping empty raw file: {site_page_path.name}")
                        try: site_page_path.unlink()
                        except OSError as del_empty_e:
                             if scraper_logger: scraper_logger.warning(f"Could not delete empty file {site_page_path.name}: {del_empty_e}")
                        continue

                    # Using the same improved block splitting as in _globally_merge_rot_site_files
                    raw_blocks_with_delimiter = page_file_content.split("--- TENDER END ---")
                    individual_tender_blocks = []
                    for block_segment in raw_blocks_with_delimiter:
                        if "--- TENDER START ---" in block_segment:
                            start_index = block_segment.find("--- TENDER START ---")
                            if start_index != -1:
                                reconstructed_block = block_segment[start_index:].strip() + "\n--- TENDER END ---"
                                individual_tender_blocks.append(reconstructed_block.strip())

                    for tender_block_text in individual_tender_blocks:
                        if not tender_block_text: continue # Skip if somehow an empty block was formed
                        # No need to check for "--- TENDER START ---" again as reconstruction ensures it
                        clean_tender_block = tender_block_text # Already stripped during reconstruction
                        
                        tender_block_hash = hash(clean_tender_block)
                        if tender_block_hash not in seen_tender_block_hashes_for_site:
                            outfile_site.write(clean_tender_block + "\n\n") # Add double newline
                            seen_tender_block_hashes_for_site.add(tender_block_hash)
                            merged_tender_blocks_for_site += 1
                    try:
                        site_page_path.unlink() # Delete the raw page file after processing
                    except OSError as del_raw_e:
                        if scraper_logger: scraper_logger.warning(f"Could not delete raw file {site_page_path.name} for site '{site_key}': {del_raw_e}")
                except Exception as e_proc_file:
                    if scraper_logger: scraper_logger.error(f"Error processing/deleting raw file {site_page_path.name} for site '{site_key}': {e_proc_file}")
        
        if scraper_logger: scraper_logger.info(f"✅ Merged {merged_tender_blocks_for_site} unique tender blocks for site '{site_key}' into: {final_site_output_path}")
        
        try: # Attempt to remove the site-specific raw page directory if it's empty
            if not any(raw_pages_dir_for_site.iterdir()):
                raw_pages_dir_for_site.rmdir()
                if scraper_logger: scraper_logger.info(f"Successfully removed empty raw directory for site '{site_key}': {raw_pages_dir_for_site}")
            else:
                if scraper_logger: scraper_logger.info(f"Raw directory for site '{site_key}' is not empty, not removed: {raw_pages_dir_for_site}")
        except OSError as rmdir_site_e:
            if scraper_logger: scraper_logger.warning(f"Could not remove raw directory {raw_pages_dir_for_site} for site '{site_key}' (it might not be empty or permission issue): {rmdir_site_e}")
            
        return merged_tender_blocks_for_site, final_site_output_path
    except Exception as e_merge_site:
        if scraper_logger: scraper_logger.error(f"Failed during merge/write for site '{site_key}': {e_merge_site}", exc_info=True)
        return merged_tender_blocks_for_site, None # Return current count and None for path

# --- Main Execution Function ---

async def run_scrape_for_one_site(
    site_key: str,
    site_config: Dict[str, str],
    global_settings: Dict[str, Any]
):
    current_site_logger = setup_site_specific_logging(site_key) # This now uses new LOGS path
    current_site_logger.info(f"🚀 Starting scrape run for site: {site_key}...")
    start_time_site = datetime.datetime.now()

    site_base_url = site_config.get("base_url")
    site_domain = site_config.get("domain")
    if not site_base_url or not site_domain:
        current_site_logger.error(f"Site configuration for '{site_key}' is missing 'base_url' or 'domain'. Skipping.")
        return

    max_pages = int(global_settings.get("scrape_py_limits", {}).get("max_list_pages", MAX_PAGES_TO_FETCH_DEFAULT))
    retry_limit = int(global_settings.get("scrape_py_limits", {}).get("retries", RETRY_LIMIT_DEFAULT))
    list_concurrency = int(global_settings.get("concurrency", {}).get("list_pages", CONCURRENCY_DEFAULT))
    detail_concurrency = int(global_settings.get("concurrency", {}).get("detail_pages", DETAIL_PAGE_CONCURRENCY_LIMIT_DEFAULT))
    page_timeout = int(global_settings.get("timeouts", {}).get("page_load", PAGE_LOAD_TIMEOUT_DEFAULT))
    detail_timeout = int(global_settings.get("timeouts", {}).get("detail_page", DETAIL_PAGE_TIMEOUT_DEFAULT))

    current_site_logger.info(f"Using settings: MaxPages={max_pages}, Retries={retry_limit}, ListConc={list_concurrency}, DetailConc={detail_concurrency}, PageTimeout={page_timeout}, DetailTimeout={detail_timeout}")

    safe_site_key_dir = re.sub(r'[^\w\-]+', '_', site_key)
    # MODIFIED: Use new base path for raw pages
    raw_pages_dir_current_site = REG_RAW_PAGES_DIR_BASE / safe_site_key_dir
    raw_pages_dir_current_site.mkdir(parents=True, exist_ok=True)

    merged_blocks_count_site = 0
    final_merged_file_path_site: Optional[Path] = None

    try:
        async with async_playwright() as p:
            # fetch_all_pages_for_site saves to raw_pages_dir_current_site
            await fetch_all_pages_for_site(
                p, site_key, site_base_url, site_domain,
                max_pages, list_concurrency, detail_concurrency,
                page_timeout, detail_timeout, retry_limit,
                raw_pages_dir_current_site
            )

        current_site_logger.info(f"Merge and cleanup phase for site {site_key}...")
        # MODIFIED: Pass REG_DATA_DIR as the root for where MergedSiteSpecific will be created by the merge function
        merged_blocks_count_site, final_merged_file_path_site = await merge_site_specific_raw_files(
            site_key, raw_pages_dir_current_site, REG_DATA_DIR # Changed from BASE_DATA_DIR_ROOT
        )

    except Exception as e_run_site:
        current_site_logger.critical(f"💥 CRITICAL ERROR during scrape run for site {site_key}: {type(e_run_site).__name__} - {e_run_site}", exc_info=True)
    finally:
        end_time_site = datetime.datetime.now()
        duration_site = end_time_site - start_time_site
        log_status_site = "completed" if merged_blocks_count_site > 0 and final_merged_file_path_site else "failed or produced no output"

        current_site_logger.info(f"🏁 Scrape run for site {site_key} {log_status_site}. Duration: {duration_site}. Merged unique tender blocks: {merged_blocks_count_site}.")
        if final_merged_file_path_site:
            current_site_logger.info(f"   Final site-specific merged file for {site_key}: {final_merged_file_path_site}")
        elif merged_blocks_count_site > 0 :
             current_site_logger.error(f"   Error: Tender blocks merged for {site_key}, but final file path not returned by merge function.")
        else:
            current_site_logger.warning(f"   No final site-specific merged file created for site {site_key}.")

# --- Standalone Execution Block (for testing scrape.py directly) ---
if __name__ == "__main__":
    print(f"--- Running {Path(__file__).name} in standalone test mode ---")
    test_site_key = "CPPP_Test" 
    test_site_config_data = {
        "base_url": "https://eprocure.gov.in/eprocure/app?component=%24TablePages.linkPage&page=FrontEndAdvancedSearchResult&service=direct&session=T&sp=AFrontEndAdvancedSearchResult%2Ctable&sp={}",
        "domain": "https://eprocure.gov.in/eprocure/app?"
    }
    test_global_settings = {
        "scrape_py_limits": {"max_list_pages": 1, "retries": 0}, # Fetch only 1 page, 0 retry for quick test
        "concurrency": {"list_pages": 1, "detail_pages": 1},     
        "timeouts": {"page_load": 60000, "detail_page": 60000}   # Reasonably short timeouts
    }

    # For testing with the provided Page.html, we would ideally mock the page.content()
    # For a live test, you'd point to a real tender detail page URL that has this structure.
    # This standalone mode is more for testing the overall flow.
    # To specifically test _scrape_detail_page with Page.html:
    # 1. Save Page.html to a file.
    # 2. Create a small async test function that reads this file, passes its content to BeautifulSoup,
    #    and then calls the relevant parts of _scrape_detail_page or the helper functions.

    print(f"To test with specific HTML like Page.html, you'd typically mock network calls or use a local file.")

    try:
        import playwright.sync_api # type: ignore
        with playwright.sync_api.sync_playwright() as p_test: # type: ignore
            browser_test = p_test.chromium.launch(headless=True)
            browser_test.close()
        print(f"Playwright check OK for standalone run of {Path(__file__).name}.")
    except Exception as play_err_test: # pragma: no cover
         print(f"Playwright check failed: {play_err_test}")
         print("Ensure browsers are installed: run 'python -m playwright install --with-deps'")
         import sys; sys.exit(1)

    try:
        asyncio.run(run_scrape_for_one_site(test_site_key, test_site_config_data, test_global_settings))
    except Exception as main_err:
         print(f"Error during standalone test run: {main_err}")

    print(f"--- Standalone test mode finished for {Path(__file__).name} ---")
