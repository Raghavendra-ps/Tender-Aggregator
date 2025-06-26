# Tender-Aggregator/utils/extract_structured_rot.py
import json
from pathlib import Path
import re
from bs4 import BeautifulSoup, Tag
import argparse 
import logging
import datetime 
from typing import Optional, Dict, List, Any

# --- Database Imports ---
# This block assumes database.py is in the parent directory.
# This setup is for when the script is called from the project root.
import sys
# Add project root to path to allow imports from parent directory
sys.path.append(str(Path(__file__).parent.parent.resolve()))
try:
    from database import SessionLocal, Tender, TenderResult, Bidder, TenderBid, db_logger
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import database models. Ensure 'database.py' is in the project root. Details: {e}")
    sys.exit(1)
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError


# --- Setup basic logging ---
logger = logging.getLogger("extract_structured_rot")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] (StructExtract) %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) 

# --- Helper Functions (Core parsing logic remains unchanged) ---
def safe_get_text(element: Optional[Tag], separator=" ", strip=True) -> Optional[str]:
    if not element: return None
    try:
        text = element.get_text(separator=separator, strip=strip)
        if text and text.lower() not in ['none', 'n/a', '-', '']:
            return text
    except: pass
    return None

def parse_amount_to_numeric(amount_text: Optional[str]) -> Optional[float]:
    if not amount_text: return None
    try:
        cleaned_text = re.sub(r'[^\d\.\-]', '', str(amount_text))
        if cleaned_text and \
           (cleaned_text.replace('.', '', 1).replace('-', '', 1).isdigit() or \
            (cleaned_text.startswith('-') and cleaned_text[1:].replace('.', '', 1).isdigit())):
            return float(cleaned_text)
    except: pass
    logger.debug(f"Could not parse amount: '{amount_text}' to float.")
    return None

def parse_document_link_cell(cell: Tag) -> Optional[Dict[str, Optional[str]]]:
    link_tag = cell.find('a', href=True)
    full_text = cell.get_text(strip=True) 
    doc_info: Dict[str, Optional[str]] = {}
    link_text_candidate = None; href = None
    if link_tag:
        link_text_candidate = safe_get_text(link_tag)
        raw_href = link_tag.get('href');
        if raw_href and raw_href.strip() != '#': href = raw_href.strip()
    final_link_text = link_text_candidate
    if not final_link_text or final_link_text.lower() == "none":
        if full_text and full_text.lower() != "none": 
            final_link_text = full_text
        elif href and not final_link_text:
            final_link_text = Path(href).name
    if not final_link_text: return None
    doc_info["text"] = final_link_text
    doc_info["link"] = href
    size_match = re.search(r'\(([\d\.]+\s*(?:KB|MB|GB|Bytes))\)', full_text, re.IGNORECASE)
    doc_info["size"] = size_match.group(1) if size_match else None
    if doc_info["text"] and size_match and doc_info["text"].endswith(size_match.group(0)):
        doc_info["text"] = doc_info["text"][:-len(size_match.group(0))].strip()
    if not doc_info.get("text") and doc_info.get("link"):
        doc_info["text"] = Path(str(doc_info["link"])).name
    if doc_info.get("text") and doc_info.get("text").lower() == "none":
         if doc_info.get("link"):
            doc_info["text"] = Path(str(doc_info["link"])).name
         else:
            doc_info["text"] = None
    return doc_info if doc_info.get("text") or doc_info.get("link") else None

def parse_html_table_to_lod(table_element: Tag, section_title_debug: str, header_row_to_skip: Optional[Tag] = None) -> List[Dict[str, Any]]:
    parsed_table_lod: List[Dict[str, Any]] = []
    headers: List[str] = []
    all_rows = table_element.find_all('tr') 
    if not all_rows: return []
    header_row_tag: Optional[Tag] = None
    for r_idx, row in enumerate(all_rows):
        if row is header_row_to_skip: continue 
        ths = row.find_all('th', recursive=False) 
        if not ths: ths = row.find_all('th')
        if ths:
            headers = [safe_get_text(th, strip=True) or f"Header_{i+1}" for i, th in enumerate(ths)]
            header_row_tag = row; break 
    if not headers and all_rows:
        for r_idx, row in enumerate(all_rows):
            if row is header_row_to_skip: continue
            if r_idx > 0 and row.find('td', class_='section_head', recursive=False) and len(row.find_all(['td','th'],recursive=False))==1: continue
            potential_header_tds = row.find_all('td', recursive=False)
            if len(potential_header_tds) > 1:
                is_header_like = any(td.find('b') for td in potential_header_tds) or any('td_caption' in td.get('class', []) for td in potential_header_tds)
                is_first_valid_row_for_header = not any(all_rows[prev_r_idx] is not header_row_to_skip and not (all_rows[prev_r_idx].find('td', class_='section_head', recursive=False) and len(all_rows[prev_r_idx].find_all(['td','th'],recursive=False))==1) for prev_r_idx in range(r_idx))
                if is_header_like or is_first_valid_row_for_header:
                    headers = [safe_get_text(td, strip=True) or f"Header_{i+1}" for i, td in enumerate(potential_header_tds)]
                    header_row_tag = row; break
    if not headers:
        first_data_row_cells = next((r.find_all(['td', 'th'], recursive=False) for r in all_rows if r is not header_row_to_skip and not (r.find('td', class_='section_head', recursive=False) and len(r.find_all(['td','th'],recursive=False)) == 1) and r.find_all(['td', 'th'], recursive=False)), None)
        if first_data_row_cells:
            headers = [f"column_{i+1}" for i in range(len(first_data_row_cells))]
            header_row_tag = None
        else: return []
    cleaned_headers = [(re.sub(r'[^a-zA-Z0-9_]', '', str(h).replace(' ', '_').replace('.', '').replace('/', '_').replace('(', '').replace(')', '').strip()) if h else f"column_{i+1}") for i, h in enumerate(headers)]
    if not any(ch for ch in cleaned_headers if ch and not ch.startswith("column_")): 
        cleaned_headers = [f"column_{i+1}" for i in range(len(headers))]
    for row in all_rows:
        if row is header_row_to_skip or (header_row_tag and row is header_row_tag): continue
        if row.find('td', class_='section_head', recursive=False) and len(row.find_all(['td','th'], recursive=False)) == 1: continue
        cells = row.find_all(['td', 'th'], recursive=False)
        if not cells or not any(safe_get_text(c) for c in cells): continue
        row_dict: Dict[str, Any] = {} 
        for i, cell in enumerate(cells):
            header_key = cleaned_headers[i] if i < len(cleaned_headers) else f"extra_column_{i+1}"
            cell_value_str = safe_get_text(cell, separator=" ")
            row_dict[header_key] = cell_value_str
            if header_key and cell_value_str and ("Value" in header_key or "Amount" in header_key or "Rate" in header_key):
                numeric_val = parse_amount_to_numeric(cell_value_str)
                if numeric_val is not None: row_dict[header_key + "_Numeric"] = numeric_val
        if row_dict: parsed_table_lod.append(row_dict)
    return parsed_table_lod

def parse_rot_html_to_structured_data(html_content: str, original_filename: str) -> dict:
    # This entire function remains the same as the improved version from our last exchange.
    # It correctly parses the HTML into a structured dictionary.
    soup = BeautifulSoup(html_content, 'html.parser')
    data: Dict[str, Any] = {"original_filename": original_filename, "source_site_key": None, "document_header": {}, "main_tender_info": {}, "key_financials": {}, "critical_dates": {}, "location_info": {}, "bids_list_data": [], "financial_evaluation_bid_list_data": [], "other_sections_kv": {}, "other_sections_table": {}}
    page_head_td = soup.find('td', class_='page_head');
    if page_head_td: data["document_header"]["system_name"] = safe_get_text(page_head_td)
    html_title_tag = soup.find('title') 
    if html_title_tag: data["document_header"]["html_page_title"] = safe_get_text(html_title_tag)
    report_title_td = soup.find('td', class_='page_title') 
    if report_title_td: data["document_header"]["report_display_title"] = safe_get_text(report_title_td)
    date_td = soup.find(lambda tag: tag.name == 'td' and tag.has_attr('class') and 'td_space' in tag['class'] and tag.string and "Date :" in tag.string.strip())
    if date_td: 
        date_text = safe_get_text(date_td)
        if date_text: data["document_header"]["report_date"] = date_text.replace("Date :", "").strip()
    main_content_container: Optional[Tag] = soup.find('form', id='bidSummaryForm')
    if not main_content_container: main_content_container = soup.body if soup.body else soup 
    if not main_content_container: return data 
    main_info_table_found: Optional[Tag] = None
    all_tables_in_content = main_content_container.find_all('table', class_='table_list')
    for candidate_table in all_tables_in_content:
        first_tr = candidate_table.find('tr')
        if first_tr and first_tr.find('td', class_='section_head'): continue
        temp_main_info_kv: Dict[str, Optional[str]] = {}
        expected_main_labels = {"organisation chain", "tender id", "tender ref no", "tender title"}
        found_labels_count = 0; is_likely_kv_structure = True
        rows_in_candidate = candidate_table.find_all('tr')
        if not rows_in_candidate: continue
        for row in rows_in_candidate:
            cols = row.find_all('td', recursive=False)
            if len(cols) == 2:
                key_raw = safe_get_text(cols[0]); value_raw = safe_get_text(cols[1])
                if key_raw:
                    key_clean = key_raw.replace(':', '').strip()
                    temp_main_info_kv[key_clean] = value_raw
                    if key_clean.lower() in expected_main_labels: found_labels_count += 1
            elif len(cols) != 0 and not row.find('td', class_='section_head'): is_likely_kv_structure = False; break
        if is_likely_kv_structure and found_labels_count >= 1: 
            data["main_tender_info"] = temp_main_info_kv
            main_info_table_found = candidate_table
            if main_info_table_found and main_info_table_found.parent: main_info_table_found.decompose()
            break 
    if not main_info_table_found: logger.warning(f"Could not identify Main Tender Info block for {original_filename}.")
    search_context_for_sections = main_content_container
    section_head_tds = search_context_for_sections.find_all('td', class_='section_head')
    for header_td in section_head_tds:
        if not header_td.parent: continue
        section_title_raw = safe_get_text(header_td)
        if not section_title_raw: continue
        table_containing_header: Optional[Tag] = header_td.find_parent('table')
        if not table_containing_header: continue
        table_for_section: Optional[Tag] = None; header_row_of_this_section: Optional[Tag] = header_td.find_parent('tr')
        all_trs_in_header_table = table_containing_header.find_all('tr', recursive=False)
        is_header_table_separate = bool(header_row_of_this_section and len(all_trs_in_header_table) == 1 and all_trs_in_header_table[0] is header_row_of_this_section)
        if is_header_table_separate:
            table_for_section = table_containing_header.find_next_sibling('table')
            if not table_for_section: continue
        else: table_for_section = table_containing_header
        if not table_for_section or not table_for_section.parent: continue
        section_title_lower = section_title_raw.lower()
        parsed_data_for_section = parse_html_table_to_lod(table_for_section, section_title_raw, header_row_to_skip=header_row_of_this_section)
        if "bids list" == section_title_lower: data["bids_list_data"] = parsed_data_for_section
        elif "financial evaluation bid list" == section_title_lower: data["financial_evaluation_bid_list_data"] = parsed_data_for_section
        else: 
            is_kv_like = len(parsed_data_for_section) > 0 and len(parsed_data_for_section[0]) == 2 and (sum(1 for row in parsed_data_for_section if len(row) == 2) / len(parsed_data_for_section) > 0.7)
            if is_kv_like:
                kv_data: Dict[str, Any] = {}
                for row_dict_item in parsed_data_for_section:
                    item_keys = list(row_dict_item.keys()); key_header = item_keys[0] if item_keys else None; value_header = item_keys[1] if len(item_keys) > 1 else None
                    if key_header and value_header:
                        key_text = str(row_dict_item.get(key_header,"")).replace(':','').strip(); value_content_str = str(row_dict_item.get(value_header,""))
                        if key_text:
                            original_value_td_for_kv = next((td_pair[1] for tr_in_section in table_for_section.find_all('tr') if (td_pair := tr_in_section.find_all('td', recursive=False)) and len(td_pair) == 2 and safe_get_text(td_pair[0],"").replace(':','').strip() == key_text), None)
                            if original_value_td_for_kv and ("document" in key_text.lower() or "chart" in key_text.lower()):
                                doc_info = parse_document_link_cell(original_value_td_for_kv)
                                kv_data[key_text] = doc_info if doc_info else value_content_str
                            else: kv_data[key_text] = value_content_str
                if kv_data: data["other_sections_kv"][section_title_raw] = kv_data
            elif parsed_data_for_section: data["other_sections_table"][section_title_raw] = parsed_data_for_section
        if table_for_section.parent: table_for_section.decompose()
        if table_containing_header and table_containing_header is not table_for_section and table_containing_header.parent: table_containing_header.decompose()
    all_kv_sources = [data.get("main_tender_info", {})] + [c for c in data.get("other_sections_kv", {}).values() if isinstance(c, dict)]
    for kv_source in all_kv_sources:
        for key, value in kv_source.items():
            if not isinstance(key, str): continue
            kl = key.lower().replace(':','').strip(); val_str = str(value) if not isinstance(value, dict) else (value.get("text") if isinstance(value.get("text"), str) else None)
            if "tender value" in kl and "tender_value_text" not in data["key_financials"]: data["key_financials"]["tender_value_text"] = val_str; data["key_financials"]["tender_value_numeric"] = parse_amount_to_numeric(val_str)
            elif ("emd amount" in kl or "e.m.d amount" in kl) and "emd_amount_text" not in data["key_financials"]: data["key_financials"]["emd_amount_text"] = val_str; data["key_financials"]["emd_amount_numeric"] = parse_amount_to_numeric(val_str)
            date_mapping_post = {"published date": "published_date", "publish date": "published_date", "document download / sale start date": "doc_download_start_date", "clarification start date": "clarification_start_date", "bid submission start date": "bid_submission_start_date", "bid submission end date": "closing_date", "document download / sale end date": "doc_download_end_date", "clarification end date": "clarification_end_date", "bid opening date": "opening_date", "financial bid opening date": "financial_bid_opening_date_actual"}
            if kl in date_mapping_post and isinstance(val_str, str) and val_str.strip() and date_mapping_post[kl] not in data["critical_dates"]: data["critical_dates"][date_mapping_post[kl]] = val_str
            if kl == "location" and isinstance(val_str, str) and val_str.strip() and "raw_location_text" not in data["location_info"]: data["location_info"]["raw_location_text"] = val_str
            if kl == "pincode" and isinstance(val_str, str) and val_str.strip() and "pincode" not in data["location_info"]: data["location_info"]["pincode"] = val_str
    title_for_cat = data.get("main_tender_info", {}).get("Tender Title") or data.get("main_tender_info", {}).get("Work Description")
    if not title_for_cat:
        for section_name, section_data in data.get("other_sections_kv", {}).items():
            if "work item details" in section_name.lower() or "work/item(s)" in section_name.lower():
                if isinstance(section_data, dict):
                    title_for_cat = section_data.get("Title") or section_data.get("Work Description") or section_data.get("Description")
                    if title_for_cat: break
    data["main_tender_info"]["Work Category (Raw Text)"] = title_for_cat if title_for_cat else None
    data["main_tender_info"]["Work Category (Classified)"] = "NEEDS_CLASSIFICATION"
    return data

# --- NEW: Database Interaction Functions ---

def parse_date(date_str: Optional[str]) -> Optional[datetime.datetime]:
    """Parses a date string with common formats into a datetime object."""
    if not date_str or not isinstance(date_str, str):
        return None
    # Formats to try, in order of preference
    formats = [
        '%d-%b-%Y %I:%M %p',  # 21-Feb-2025 02:32 PM
        '%d-%b-%Y',          # 21-Feb-2025
    ]
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    logger.warning(f"Could not parse date string: '{date_str}'")
    return None

def get_or_create_bidder(db: Session, bidder_name: str) -> Bidder:
    """Gets a bidder from the DB or creates a new one if it doesn't exist."""
    bidder = db.query(Bidder).filter(Bidder.bidder_name == bidder_name).first()
    if not bidder:
        db_logger.info(f"Creating new bidder: '{bidder_name}'")
        bidder = Bidder(bidder_name=bidder_name)
        db.add(bidder)
        db.flush() # Flush to get the ID for relationships
    return bidder

def save_rot_data_to_db(db: Session, structured_data: Dict[str, Any], site_key: str):
    """Saves the structured ROT data into the database, updating relevant tables."""
    main_info = structured_data.get("main_tender_info", {})
    tender_id_str = main_info.get("Tender ID")
    
    if not tender_id_str:
        logger.error("No 'Tender ID' found in parsed data. Cannot save to database.")
        return

    # Find the main tender record
    tender = db.query(Tender).filter(Tender.tender_id == tender_id_str).first()
    
    if not tender:
        # This is an edge case: we have the result but not the original tender notice.
        # We create a placeholder Tender record.
        db_logger.warning(f"No existing tender found for ID '{tender_id_str}'. Creating placeholder record.")
        tender = Tender(
            tender_id=tender_id_str,
            source_site=site_key,
            tender_title=main_info.get("Tender Title"),
            organisation_chain=main_info.get("Organisation Chain"),
            status="Result Announced (Placeholder)"
        )
        db.add(tender)
        db.flush() # Ensure the tender object gets its primary key 'id'
    else:
        # Update the status of the existing tender record
        tender.status = "Result Announced"
        db_logger.info(f"Updating tender '{tender_id_str}' status to 'Result Announced'.")

    # Create or update the TenderResult record
    tender_result = db.query(TenderResult).filter(TenderResult.tender_id_fk == tender.id).first()
    if not tender_result:
        tender_result = TenderResult(tender=tender)
        db.add(tender_result)
    
    tender_result.final_stage = next((s for s in structured_data.get("other_sections_kv", {}) if "evaluation summary" in s.lower()), "Unknown")
    tender_result.full_summary_json = structured_data
    
    # Process Bidders and Bids
    # We combine both bid lists as they refer to the same set of bidders
    bids_list = structured_data.get("bids_list_data", [])
    fin_eval_list = structured_data.get("financial_evaluation_bid_list_data", [])

    # Create a dictionary to merge bid info by bidder name
    merged_bids = {}
    for bid in bids_list:
        bidder_name = bid.get("Bidder_Name")
        if bidder_name:
            merged_bids[bidder_name] = {
                "status": bid.get("Status"),
                "remarks": bid.get("Remarks")
            }
            
    for fin_bid in fin_eval_list:
        bidder_name = fin_bid.get("Bidder_Name")
        if bidder_name:
            if bidder_name not in merged_bids:
                merged_bids[bidder_name] = {}
            merged_bids[bidder_name].update({
                "rank": fin_bid.get("Rank"),
                "value": fin_bid.get("Value_Numeric")
            })

    # Now, iterate through the merged bid data and create DB records
    for bidder_name, bid_details in merged_bids.items():
        bidder_record = get_or_create_bidder(db, bidder_name)
        
        # Check if a bid for this tender and bidder already exists to avoid duplicates
        existing_bid = db.query(TenderBid).filter(
            TenderBid.tender_id_fk == tender.id,
            TenderBid.bidder_id_fk == bidder_record.id
        ).first()

        if existing_bid:
            # Update existing bid
            existing_bid.bid_status = bid_details.get("status")
            existing_bid.bid_rank = bid_details.get("rank")
            existing_bid.bid_value = bid_details.get("value")
            db_logger.debug(f"Updating existing bid for tender '{tender_id_str}' and bidder '{bidder_name}'.")
        else:
            # Create new bid
            new_bid = TenderBid(
                tender=tender,
                bidder=bidder_record,
                bid_status=bid_details.get("status"),
                bid_rank=bid_details.get("rank"),
                bid_value=bid_details.get("value")
            )
            db.add(new_bid)
            db_logger.debug(f"Creating new bid for tender '{tender_id_str}' and bidder '{bidder_name}'.")

# --- MODIFIED: Main execution logic ---

def main(html_file_path: Path, site_key: str):
    """
    Main function to parse HTML and save results to the database.
    """
    # --- NEW: Ensure DB is initialized before using it ---
    try:
        # We're importing init_db from database.py now
        from database import init_db
        init_db()
    except Exception as e:
        logger.critical(f"Failed to initialize database from extract_structured_rot.py. Error: {e}")
        return
    # --- END NEW ---

    if not html_file_path.is_file():
        logger.error(f"Input HTML file not found: {html_file_path}")
        return

    logger.info(f"Processing ROT HTML file: {html_file_path.name}")
    try:
        with open(html_file_path, 'r', encoding='utf-8', errors='replace') as f:
            html_content = f.read()
    except Exception as e:
        logger.error(f"Could not read HTML file {html_file_path}: {e}")
        return

    # 1. Parse the HTML content into a structured dictionary
    structured_data = parse_rot_html_to_structured_data(html_content, html_file_path.name)
    structured_data["source_site_key"] = site_key # Ensure site_key is in the data

    if not structured_data.get("main_tender_info", {}).get("Tender ID"):
        logger.error(f"Parsing failed to extract Tender ID from {html_file_path.name}. Aborting database operation.")
        return

    # 2. Save the structured data to the database
    db: Session = SessionLocal()
    try:
        save_rot_data_to_db(db, structured_data, site_key)
        db.commit()
        logger.info(f"Successfully saved data for Tender ID '{structured_data['main_tender_info']['Tender ID']}' to the database.")
    except SQLAlchemyError as e:
        db.rollback()
        logger.critical(f"Database error while processing {html_file_path.name}: {e}", exc_info=True)
    except Exception as e:
        db.rollback()
        logger.critical(f"An unexpected error occurred during database operation for {html_file_path.name}: {e}", exc_info=True)
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse ROT HTML summary file and save to database.")
    parser.add_argument("html_file_path", type=Path, help="Path to the ROT HTML summary file.")
    parser.add_argument("--site-key", type=str, required=True, help="Site key (e.g., Goa). Used for data field.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (DEBUG level).")
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        db_logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        db_logger.setLevel(logging.INFO)

    main(html_file_path=args.html_file_path, site_key=args.site_key)
