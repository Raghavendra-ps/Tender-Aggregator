# Tender-Aggregator/utils/extract_structured_rot.py
import json
from pathlib import Path
import re
from typing import Optional, Dict, List
from bs4 import BeautifulSoup, Tag, NavigableString
import argparse
import logging
import datetime # For potential date parsing

# --- Setup basic logging for this script ---
logger = logging.getLogger("extract_structured_rot")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] (StructExtract) %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Default, can be overridden by -v

# --- Helper Functions ---
def safe_get_text(element: Optional[Tag], separator=" ", strip=True) -> Optional[str]:
    if not element:
        return None
    try:
        text = element.get_text(separator=separator, strip=strip)
        return text if text else None
    except Exception:
        return None

def parse_amount_to_numeric(amount_text: Optional[str]) -> Optional[float]:
    if not amount_text:
        return None
    try:
        # Remove currency symbols, commas, and strip whitespace
        cleaned_text = re.sub(r'[₹,]', '', amount_text).strip()
        if cleaned_text:
            return float(cleaned_text)
    except ValueError:
        logger.debug(f"Could not parse '{amount_text}' to float.")
    return None

def parse_document_link_cell(cell: Tag) -> Optional[Dict[str, str]]:
    """Parses a cell known to contain a document link and size."""
    link_tag = cell.find('a', href=True)
    full_text = cell.get_text(strip=True)
    doc_info = {}
    if link_tag:
        doc_info["text"] = link_tag.get_text(strip=True)
        doc_info["link"] = link_tag.get('href', '#') # Keep relative link for now
    elif full_text and full_text.lower() != "none":
        doc_info["text"] = full_text # Fallback if no <a> but text exists
        doc_info["link"] = "#"
    else:
        return None # No meaningful document info

    size_match = re.search(r'\(([\d\.]+\s*(KB|MB|GB|Bytes))\)', full_text, re.IGNORECASE)
    doc_info["size"] = size_match.group(1) if size_match else "N/A"
    
    # Remove size from main text if present and link text was just the filename
    if doc_info["text"] and size_match and doc_info["text"].endswith(size_match.group(0)):
        doc_info["text"] = doc_info["text"][:-len(size_match.group(0))].strip()

    return doc_info if doc_info.get("text") else None


# --- Main Parsing Logic ---
def parse_rot_html_to_structured_data(html_content: str, original_filename: str) -> dict:
    """
    Main parsing function. Adapts logic from dashboard.py's parse_rot_summary_html
    to produce a structured dictionary as defined for competitor analysis.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    data = {
        "original_filename": original_filename,
        "source_site_key": None, # To be populated by the caller if known
        "document_header": {},
        "main_tender_info": {},
        "key_financials": {},
        "critical_dates": {}, # This might be part of main_tender_info or a section
        "location_info": {},  # This might be part of main_tender_info or a section
        "sections_kv": {},  # For sections parsed as key-value
        "sections_table": {} # For sections parsed as tables
    }

    # 1. Document Header
    page_head_td = soup.find('td', class_='page_head')
    if page_head_td: data["document_header"]["system_name"] = safe_get_text(page_head_td)
    page_title_td = soup.find('td', class_='page_title')
    if page_title_td: data["document_header"]["report_title"] = safe_get_text(page_title_td)
    date_td_space = soup.find('td', class_='td_space', string=re.compile(r'Date\s*:')) # Use string for exact match
    if date_td_space: data["document_header"]["report_date"] = safe_get_text(date_td_space).replace("Date :", "").strip()

    # 2. Main Content Container (crucial for scoping finds)
    main_content_container = soup.find('form', id='bidSummaryForm')
    if not main_content_container: main_content_container = soup.find('div', class_='border')
    if not main_content_container: main_content_container = soup.body if soup.body else soup # Fallback

    if not main_content_container:
        logger.warning(f"No main content container found for {original_filename}. Parsing may be incomplete.")
        return data # Return partially filled data

    # 3. Main Tender Info Block (typically the first table.table_list)
    # This needs to be robust. We assume it's key-value.
    main_info_table = main_content_container.find('table', class_='table_list')
    if main_info_table:
        is_main_info_block = False
        expected_labels = ["organisation chain", "tender id", "tender ref no", "tender title"]
        found_labels_count = 0
        current_main_info_kv = {}

        for row in main_info_table.find_all('tr', class_='td_caption', recursive=False):
            cols = row.find_all('td', recursive=False)
            if len(cols) == 2:
                key_raw = safe_get_text(cols[0])
                value_raw = safe_get_text(cols[1])
                if key_raw:
                    key_clean = key_raw.replace(':', '').strip()
                    current_main_info_kv[key_clean] = value_raw
                    if key_clean.lower() in expected_labels:
                        found_labels_count += 1
        
        if found_labels_count >= 2: # Heuristic: if it looks like the main info block
            data["main_tender_info"] = current_main_info_kv
            logger.debug(f"Extracted main_tender_info for {original_filename}")
            # Decompose after processing to avoid it being re-parsed by general section logic
            if main_info_table.parent: main_info_table.decompose()
        else:
            logger.debug(f"First table_list not identified as main_tender_info for {original_filename}")
            # Don't decompose if not main info, it might be a regular section table

    # 4. Process Sections under <td class="section_head">
    # Re-find main_content_container if it was decomposed, or use original soup if it was the container
    if not main_content_container.parent: main_content_container = soup # If it was decomposed

    section_head_tds = main_content_container.find_all('td', class_='section_head')
    processed_section_tables = set() # To avoid double-processing tables

    for header_td in section_head_tds:
        section_title_raw = safe_get_text(header_td)
        if not section_title_raw: continue
        
        logger.debug(f"Processing section titled: '{section_title_raw}'")

        # Determine the table for this section
        # This logic needs to be robust, similar to your dashboard parser
        table_for_section = None
        parent_table_of_header = header_td.find_parent('table')
        if parent_table_of_header:
            # If header is a full-width row in its own table, the data is likely in the *next* table
            header_row = header_td.find_parent('tr')
            if header_row and len(header_row.find_all(['td','th'], recursive=False)) == 1 and header_td.has_attr('colspan'):
                next_table_sibling = parent_table_of_header.find_next_sibling('table')
                if next_table_sibling:
                    table_for_section = next_table_sibling
                else: # Header table was alone, use itself (might just be header)
                    table_for_section = parent_table_of_header
            else: # Header is a cell within a larger table that contains the data
                table_for_section = parent_table_of_header
        
        if not table_for_section or id(table_for_section) in processed_section_tables:
            logger.debug(f"Skipping section '{section_title_raw}', table not found or already processed.")
            continue
        
        processed_section_tables.add(id(table_for_section))

        # Classify table as key-value or general table and parse
        rows = table_for_section.find_all('tr')
        is_kv_like_section = False
        if len(rows) > 0:
            kv_row_count = 0; data_row_count = 0
            for r_idx, row_check in enumerate(rows):
                if row_check == header_row: continue # Skip the header row itself
                cells_check = row_check.find_all('td', recursive=False)
                if not cells_check: continue
                data_row_count +=1
                if len(cells_check) == 2: kv_row_count += 1
            if data_row_count > 0 and (kv_row_count / data_row_count) >= 0.7: # 70% threshold
                is_kv_like_section = True
        
        if is_kv_like_section:
            kv_data = {}
            for row in rows:
                if row == header_row: continue
                cells = row.find_all('td', recursive=False)
                if len(cells) == 2:
                    key = safe_get_text(cells[0]).replace(':', '').strip()
                    # Value extraction: try to get clean text, look for links if key is "Document"
                    val_cell = cells[1]
                    if key and key.lower() == "document":
                        doc_info = parse_document_link_cell(val_cell)
                        if doc_info: kv_data[key] = doc_info
                        else: kv_data[key] = safe_get_text(val_cell)
                    elif key and key.lower() == "boq comparative chart": # Special case for BOQ link
                        link_tag = val_cell.find('a', href=True)
                        if link_tag:
                            kv_data[key] = {"text": safe_get_text(link_tag), "link": link_tag.get('href')}
                        else: kv_data[key] = safe_get_text(val_cell)
                    elif key:
                         kv_data[key] = safe_get_text(val_cell, separator=" ") # Preserve spaces in values
            if kv_data: data["sections_kv"][section_title_raw] = kv_data
            logger.debug(f"Parsed section '{section_title_raw}' as key-value.")
        else: # General table
            table_content = []
            # Attempt to find headers (th or first row td.td_caption)
            headers = []
            first_tr_in_table = table_for_section.find('tr')
            if first_tr_in_table:
                header_ths = first_tr_in_table.find_all('th')
                if header_ths: headers = [safe_get_text(th) for th in header_ths]
                else:
                    header_tds = first_tr_in_table.find_all('td', class_='td_caption')
                    if header_tds and len(header_tds) > 1 : # Check if it looks like a header row
                        headers = [safe_get_text(td) for td in header_tds]
            
            if headers and any(h is not None for h in headers): # Add headers if found
                 table_content.append(headers)

            for row in rows:
                if row == header_row: continue
                if headers and row == first_tr_in_table: continue # Skip if it was the header row already added

                # Skip other nested section_head rows within this table that aren't the main one
                if row.find('td', class_='section_head') and row.find('td', class_='section_head') != header_td:
                    continue

                cells = row.find_all(['td', 'th'], recursive=False)
                if not cells: continue
                
                row_data = [safe_get_text(cell, separator=" ") for cell in cells]
                if any(cell_data for cell_data in row_data):
                    table_content.append(row_data)
            
            if table_content: data["sections_table"][section_title_raw] = table_content
            logger.debug(f"Parsed section '{section_title_raw}' as table.")
        
        # Decompose the processed table to avoid issues if main_content_container.get_text() is used later
        if table_for_section.parent and table_for_section != main_content_container:
            table_for_section.decompose()


    # --- Post-processing: Move specific known fields to dedicated top-level keys ---
    # Example: Tender Value, EMD from a 'Fee Details' like section
    fee_section_titles = [k for k in data["sections_kv"] if "fee details" in k.lower()]
    if fee_section_titles:
        fee_details_kv = data["sections_kv"][fee_section_titles[0]] # Assuming first match
        # Tender Value
        tv_keys = [k for k in fee_details_kv if "tender value" in k.lower()]
        if tv_keys: 
            data["key_financials"]["tender_value_text"] = fee_details_kv[tv_keys[0]]
            data["key_financials"]["tender_value_numeric"] = parse_amount_to_numeric(fee_details_kv[tv_keys[0]])
        # EMD Amount
        emd_keys = [k for k in fee_details_kv if "emd amount" in k.lower()]
        if emd_keys:
            data["key_financials"]["emd_amount_text"] = fee_details_kv[emd_keys[0]]
            data["key_financials"]["emd_amount_numeric"] = parse_amount_to_numeric(fee_details_kv[emd_keys[0]])
    
    # Location from Main Info or Work Item Details section
    loc_keys = ["Location", "location", "Pincode", "pincode"]
    for key, value in data.get("main_tender_info", {}).items():
        if key in loc_keys:
            data["location_info"][key.lower()] = value
    
    work_item_section_titles = [k for k in data["sections_kv"] if "work item details" in k.lower() or "work/item(s)" in k.lower()]
    if work_item_section_titles:
        work_details_kv = data["sections_kv"][work_item_section_titles[0]]
        for key, value in work_details_kv.items():
            if key.lower() in ["location", "pincode"]:
                 data["location_info"][key.lower()] = value
            if key.lower() in ["title", "work title", "work description", "description"]: # Populate main_tender_info if not already
                if not data["main_tender_info"].get("Tender Title", "").strip() and "title" in key.lower():
                    data["main_tender_info"]["Tender Title (from Work Item)"] = value
                if not data["main_tender_info"].get("Work Description", "").strip() and "description" in key.lower():
                    data["main_tender_info"]["Work Description"] = value


    # TODO: Classify "Work Category" from tender title/description (potentially an LLM call here if complex)
    # For now, placeholder:
    data["main_tender_info"]["Work Category"] = "To be classified"


    logger.info(f"Finished parsing {original_filename}.")
    return data


def main():
    parser = argparse.ArgumentParser(description="Parse ROT HTML summary file to structured JSON.")
    parser.add_argument("html_file_path", type=Path, help="Path to the ROT HTML summary file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save JSON output. If None, prints to stdout.")
    parser.add_argument("--site-key", type=str, default=None, help="Site key (e.g., state name) to add to JSON. If not provided, tries to infer from path.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (DEBUG level).")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO) # Ensure logger respects verbosity

    html_file = args.html_file_path.resolve()
    if not html_file.is_file():
        logger.error(f"Input HTML file not found: {html_file}")
        return

    try:
        with open(html_file, 'r', encoding='utf-8', errors='replace') as f:
            html_content = f.read()
    except Exception as e:
        logger.error(f"Could not read HTML file {html_file}: {e}")
        return

    structured_output = parse_rot_html_to_structured_data(html_content, html_file.name)

    if structured_output:
        # Determine site_key
        site_key_to_use = args.site_key
        if not site_key_to_use:
            # Try to infer from path: /path/to/DetailHtmls/SITE_KEY/file.html
            try:
                # Assuming HTML file is inside a SITE_KEY directory which is inside DetailHtmls
                if html_file.parent.parent.name.lower() == "detailhtmls":
                    site_key_to_use = html_file.parent.name
            except IndexError:
                logger.warning("Could not infer site_key from path structure.")
        
        if site_key_to_use:
            structured_output["source_site_key"] = site_key_to_use
            logger.debug(f"Using site_key: {site_key_to_use}")
        else:
            logger.warning("Site key not provided and could not be inferred. 'source_site_key' will be null.")


        if args.output_dir:
            output_base_dir = args.output_dir.resolve()
            # Create site-specific subdirectory if site_key is known
            if site_key_to_use:
                final_output_dir = output_base_dir / site_key_to_use
            else:
                final_output_dir = output_base_dir / "unknown_site"
            
            final_output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = html_file.stem + "_structured.json" # e.g. ROT_..._StageSummary_structured.json
            output_file_path = final_output_dir / output_filename
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_output, f, indent=2, ensure_ascii=False) # Use indent=2 for readability
                logger.info(f"Structured JSON saved to: {output_file_path}")
            except Exception as e:
                logger.error(f"Could not write JSON to file {output_file_path}: {e}")
        else:
            # Pretty print JSON to stdout
            print(json.dumps(structured_output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
