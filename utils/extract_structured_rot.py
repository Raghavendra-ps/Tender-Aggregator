# Tender-Aggregator/utils/extract_structured_rot.py
import json
from pathlib import Path
import re
from bs4 import BeautifulSoup, Tag, NavigableString 
import argparse 
import logging
import datetime 
from typing import Optional, Dict, List, Any, Union

# --- Setup basic logging ---
logger = logging.getLogger("extract_structured_rot")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] (StructExtract) %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) 

# --- Helper Functions ---
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
    if not final_link_text or final_link_text.lower() == "none": # If link text is 'None' or empty
        # Use full_text only if it's not also 'None' or empty
        if full_text and full_text.lower() != "none": 
            final_link_text = full_text
        # If link_tag existed but text was 'None', try to use href as text if final_link_text is still None
        elif href and not final_link_text:
            final_link_text = Path(href).name # Use filename from href as text

    if not final_link_text: return None # Still no meaningful text

    doc_info["text"] = final_link_text
    doc_info["link"] = href

    size_match = re.search(r'\(([\d\.]+\s*(?:KB|MB|GB|Bytes))\)', full_text, re.IGNORECASE)
    doc_info["size"] = size_match.group(1) if size_match else None
    
    # Clean size from display text if it was appended
    if doc_info["text"] and size_match and doc_info["text"].endswith(size_match.group(0)):
        doc_info["text"] = doc_info["text"][:-len(size_match.group(0))].strip()
    # If after cleaning, text is empty but link exists, use filename from link
    if not doc_info.get("text") and doc_info.get("link"):
        doc_info["text"] = Path(str(doc_info["link"])).name
    
    if doc_info.get("text") and doc_info["text"].lower() == "none": # If final text is "None"
         if doc_info.get("link"): # but there's a link, prefer filename from link
            doc_info["text"] = Path(str(doc_info["link"])).name
         else:
            doc_info["text"] = None


    return doc_info if doc_info.get("text") or doc_info.get("link") else None

def parse_html_table_to_lod(table_element: Tag, section_title_debug: str, header_row_to_skip: Optional[Tag] = None) -> List[Dict[str, Optional[str]]]:
    parsed_table_lod: List[Dict[str, Optional[str]]] = []
    headers: List[str] = []
    
    all_rows = table_element.find_all('tr') 
    if not all_rows:
        logger.debug(f"Table in section '{section_title_debug}' has no <tr> elements.")
        return []

    header_row_tag: Optional[Tag] = None
    
    # Try to find headers from <th> elements first
    for r_idx, row in enumerate(all_rows):
        if row is header_row_to_skip: continue 
        ths = row.find_all('th', recursive=False) 
        if not ths: ths = row.find_all('th') # Check deeper if not direct children
        if ths:
            headers = [safe_get_text(th, strip=True) or f"Header_{i+1}" for i, th in enumerate(ths)]
            header_row_tag = row; break 
    
    # If no <th>, try first non-skipped data row with multiple <td>s (often class 'td_caption' or containing <b>)
    if not headers and all_rows:
        for r_idx, row in enumerate(all_rows):
            if row is header_row_to_skip: continue
            if r_idx > 0 and row.find('td', class_='section_head', recursive=False) and len(row.find_all(['td','th'],recursive=False))==1: continue

            potential_header_tds = row.find_all('td', recursive=False)
            if len(potential_header_tds) > 1: 
                # Heuristic: if cells look like headers or if it's the first valid data row
                is_header_like = any(td.find('b') for td in potential_header_tds) or \
                                 any('td_caption' in td.get('class', []) for td in potential_header_tds)
                
                is_first_valid_row_for_header = True
                for prev_r_idx in range(r_idx):
                    if all_rows[prev_r_idx] is not header_row_to_skip and \
                       not (all_rows[prev_r_idx].find('td', class_='section_head', recursive=False) and \
                            len(all_rows[prev_r_idx].find_all(['td','th'],recursive=False))==1):
                        is_first_valid_row_for_header = False; break
                
                if is_header_like or is_first_valid_row_for_header:
                    headers = [safe_get_text(td, strip=True) or f"Header_{i+1}" for i, td in enumerate(potential_header_tds)]
                    header_row_tag = row; break
    
    if not headers: # Fallback to generic column names if still no headers
        if all_rows:
            first_data_row_cells = None
            for r_fr in all_rows:
                if r_fr is header_row_to_skip: continue
                if r_fr.find('td', class_='section_head', recursive=False) and len(r_fr.find_all(['td','th'],recursive=False)) == 1: continue
                cells = r_fr.find_all(['td', 'th'], recursive=False)
                if cells : first_data_row_cells = cells; header_row_tag = None; break 
            if first_data_row_cells:
                headers = [f"column_{i+1}" for i in range(len(first_data_row_cells))]
            else: logger.warning(f"Could not determine headers for table in '{section_title_debug}'."); return []
        else: return []

    cleaned_headers = [(re.sub(r'[^a-zA-Z0-9_]', '', str(h).replace(' ', '_').replace('.', '').replace('/', '_').replace('(', '').replace(')', '').strip()) if h else f"column_{i+1}") for i, h in enumerate(headers)]
    if not any(ch for ch in cleaned_headers if ch and not ch.startswith("column_")): 
        cleaned_headers = [f"column_{i+1}" for i in range(len(headers))]

    for row in all_rows:
        if row is header_row_to_skip: continue
        if header_row_tag and row is header_row_tag: continue
        if row.find('td', class_='section_head', recursive=False) and len(row.find_all(['td','th'], recursive=False)) == 1: continue

        cells = row.find_all(['td', 'th'], recursive=False)
        if not cells or not any(safe_get_text(c) for c in cells): continue

        row_dict: Dict[str, Any] = {} 
        for i, cell in enumerate(cells):
            header_key = cleaned_headers[i] if i < len(cleaned_headers) else f"extra_column_{i+1}"
            cell_value_str = safe_get_text(cell, separator=" ") # Keep spaces within cell text
            row_dict[header_key] = cell_value_str
            
            if header_key and cell_value_str and ("Value" in header_key or "Amount" in header_key or "Rate" in header_key):
                numeric_val = parse_amount_to_numeric(cell_value_str)
                if numeric_val is not None: row_dict[header_key + "_Numeric"] = numeric_val
        if row_dict: parsed_table_lod.append(row_dict)
            
    return parsed_table_lod

# --- Main Parsing Logic ---
def parse_rot_html_to_structured_data(html_content: str, original_filename: str) -> dict:
    soup = BeautifulSoup(html_content, 'html.parser')
    
    data: Dict[str, Any] = {
        "original_filename": original_filename, "source_site_key": None, 
        "document_header": {}, "main_tender_info": {}, "key_financials": {},
        "critical_dates": {}, "location_info": {},
        "bids_list_data": [], "financial_evaluation_bid_list_data": [],
        "other_sections_kv": {}, "other_sections_table": {}
    }

    # 1. Document Header - Parse from the TOP-LEVEL soup object
    page_head_td = soup.find('td', class_='page_head')
    if page_head_td: data["document_header"]["system_name"] = safe_get_text(page_head_td)
    
    html_title_tag = soup.find('title') 
    if html_title_tag: data["document_header"]["html_page_title"] = safe_get_text(html_title_tag)

    report_title_td = soup.find('td', class_='page_title') 
    if report_title_td: data["document_header"]["report_display_title"] = safe_get_text(report_title_td)
    
    date_td = soup.find(lambda tag: tag.name == 'td' and tag.has_attr('class') and \
                                     'td_space' in tag['class'] and \
                                     tag.string and "Date :" in tag.string.strip())
    if date_td: 
        date_text = safe_get_text(date_td)
        if date_text: data["document_header"]["report_date"] = date_text.replace("Date :", "").strip()
    logger.debug(f"Document Header Data: {data['document_header']}")

    # 2. Main Content Container
    main_content_container: Optional[Tag] = soup.find('form', id='bidSummaryForm')
    if not main_content_container: 
        logger.debug(f"Container 'form#bidSummaryForm' not found. Trying soup.body for {original_filename}.")
        main_content_container = soup.body if soup.body else soup 
    if not main_content_container:
        logger.error(f"CRITICAL: No usable main content container found for {original_filename}.")
        return data 

    # 3. Main Tender Info Block
    main_info_table_found: Optional[Tag] = None
    # Look for the first table with class 'table_list' that appears to be key-value main info.
    # This table is usually not directly preceded by a section_head of its own.
    all_tables_in_content = main_content_container.find_all('table', class_='table_list')
    logger.debug(f"Found {len(all_tables_in_content)} table.table_list candidates for main_info and sections.")

    for candidate_table in all_tables_in_content:
        # Heuristic: if this table itself starts with a section_head row, it's not the main info table.
        first_tr = candidate_table.find('tr')
        if first_tr and first_tr.find('td', class_='section_head'):
            logger.debug("Skipping candidate for main_info as it starts with a section_head.")
            continue 
        
        # Another heuristic: main info table usually has rows with class 'td_caption' for key-value.
        # And usually does not have an ID like 'bidValidTableView' or 'table_list' (which is for fin_eval_bid_list).
        if candidate_table.get('id') in ['bidValidTableView', 'table_list']: # Check specific table IDs
            # The financial evaluation bid list table has id="table_list" in your example.
            # The main info table usually does not have an ID, or a different one.
            logger.debug(f"Skipping candidate table with id='{candidate_table.get('id')}' as likely a data section table.")
            continue

        temp_main_info_kv: Dict[str, Optional[str]] = {}
        expected_main_labels = {"organisation chain", "tender id", "tender ref no", "tender title"}
        found_labels_count = 0
        is_likely_kv_structure = True
        
        rows_in_candidate = candidate_table.find_all('tr', class_='td_caption') # Main info often uses td_caption
        if not rows_in_candidate: # Fallback if no td_caption directly under this table_list
            rows_in_candidate = candidate_table.find_all('tr') # Check all rows then

        if not rows_in_candidate: continue

        for row in rows_in_candidate:
            cols = row.find_all('td', recursive=False)
            if len(cols) == 2:
                key_raw = safe_get_text(cols[0]); value_raw = safe_get_text(cols[1])
                if key_raw:
                    key_clean = key_raw.replace(':', '').strip()
                    temp_main_info_kv[key_clean] = value_raw
                    if key_clean.lower() in expected_main_labels: found_labels_count += 1
            elif len(cols) != 0 and not row.find_all('th'): 
                if not row.find('td', class_='section_head'):
                    is_likely_kv_structure = False; break
        
        if is_likely_kv_structure and found_labels_count >= 1: 
            data["main_tender_info"] = temp_main_info_kv
            main_info_table_found = candidate_table
            logger.info(f"Extracted main_tender_info for {original_filename} (found {found_labels_count} labels).")
            if main_info_table_found.parent: main_info_table_found.decompose()
            break 
    
    if not main_info_table_found:
        logger.warning(f"Could not identify Main Tender Info block for {original_filename}.")

    # 4. Process Sections
    search_context_for_sections = soup.find('form', id='bidSummaryForm') 
    if not search_context_for_sections : search_context_for_sections = soup.body if soup.body else soup
    if not search_context_for_sections : return data

    section_head_tds = search_context_for_sections.find_all('td', class_='section_head')
    logger.debug(f"Re-found {len(section_head_tds)} section_head elements for parsing sections.")
    
    for header_td in section_head_tds:
        if not header_td.parent: continue # Already processed and decomposed
        section_title_raw = safe_get_text(header_td)
        if not section_title_raw: continue
        logger.debug(f"Processing section: '{section_title_raw}'")

        table_containing_header: Optional[Tag] = header_td.find_parent('table')
        if not table_containing_header: logger.warning(f"S_H '{section_title_raw}' not in table."); continue

        table_for_section: Optional[Tag] = None
        header_row_of_this_section: Optional[Tag] = header_td.find_parent('tr')
        
        is_header_table_separate = False
        if header_row_of_this_section:
            all_trs_in_header_table = table_containing_header.find_all('tr', recursive=False)
            if len(all_trs_in_header_table) == 1 and \
               all_trs_in_header_table[0] is header_row_of_this_section and \
               header_td.has_attr('colspan') and \
               len(header_row_of_this_section.find_all(['td','th'],recursive=False)) == 1:
                is_header_table_separate = True

        if is_header_table_separate:
            table_for_section = table_containing_header.find_next_sibling('table', class_='table_list')
            if not table_for_section: table_for_section = table_containing_header.find_next_sibling('table')
            if table_for_section: logger.debug(f"Sec '{section_title_raw}': Data in next sibling table.")
            else: logger.warning(f"Sec '{section_title_raw}': Header separate, no next sibling. Trying parent table."); table_for_section = table_containing_header
        else:
            table_for_section = table_containing_header
            logger.debug(f"Sec '{section_title_raw}': Header within data table.")
            
        if not table_for_section: logger.warning(f"No data table for '{section_title_raw}'."); continue
        if not table_for_section.parent: logger.debug(f"Table for sec '{section_title_raw}' already decomposed. Skipping."); continue

        section_title_lower = section_title_raw.lower()
        parsed_data_for_section = parse_html_table_to_lod(table_for_section, section_title_raw, header_row_to_skip=header_row_of_this_section)

        if not parsed_data_for_section: logger.warning(f"No data parsed for section '{section_title_raw}'.")
        elif "bids list" == section_title_lower: data["bids_list_data"] = parsed_data_for_section
        elif "financial evaluation bid list" == section_title_lower: data["financial_evaluation_bid_list_data"] = parsed_data_for_section
        else: 
            original_rows = table_for_section.find_all('tr')
            kv_count_in_orig = 0; data_row_count_in_orig = 0
            for orig_row in original_rows:
                if orig_row is header_row_of_this_section: continue
                sub_header = orig_row.find('td', class_='section_head')
                if sub_header and len(orig_row.find_all('td',recursive=False)) == 1 and sub_header.has_attr('colspan'): continue
                
                orig_cells = orig_row.find_all('td', recursive=False)
                if not orig_cells: continue
                data_row_count_in_orig += 1
                if len(orig_cells) == 2: kv_count_in_orig +=1
            
            is_kv_like = (data_row_count_in_orig > 0 and (kv_count_in_orig / data_row_count_in_orig) > 0.60)

            if is_kv_like:
                kv_data: Dict[str, Any] = {}
                for row_dict_item in parsed_data_for_section:
                    item_keys = list(row_dict_item.keys())
                    key_header = item_keys[0] if item_keys else None
                    value_header = item_keys[1] if len(item_keys) > 1 else None
                    if key_header and value_header:
                        key_text = str(row_dict_item.get(key_header,"")).replace(':','').strip()
                        value_content_str = str(row_dict_item.get(value_header,"")) # Get value as string
                        if key_text: 
                            original_value_td_for_kv : Optional[Tag] = None
                            for tr_in_section in table_for_section.find_all('tr'):
                                td_pair = tr_in_section.find_all('td', recursive=False)
                                if len(td_pair) == 2:
                                    current_key_text_from_orig_table = safe_get_text(td_pair[0],"").replace(':','').strip()
                                    if current_key_text_from_orig_table == key_text:
                                        original_value_td_for_kv = td_pair[1]; break
                            if original_value_td_for_kv and ("document" in key_text.lower() or "chart" in key_text.lower()):
                                doc_info = parse_document_link_cell(original_value_td_for_kv)
                                kv_data[key_text] = doc_info if doc_info else value_content_str
                            else: kv_data[key_text] = value_content_str
                if kv_data: data["other_sections_kv"][section_title_raw] = kv_data
            else:
                if parsed_data_for_section: data["other_sections_table"][section_title_raw] = parsed_data_for_section
        
        if table_for_section.parent: table_for_section.decompose()
        if table_containing_header and table_containing_header is not table_for_section and table_containing_header.parent:
            table_containing_header.decompose()

    # --- Post-processing ---
    all_kv_sources = [data.get("main_tender_info", {})]
    for section_title, kv_content in data.get("other_sections_kv", {}).items():
        if isinstance(kv_content, dict): all_kv_sources.append(kv_content)
    logger.debug(f"Collected {len(all_kv_sources)} KV sources for post-processing.")

    for kv_source in all_kv_sources:
        for key, value in kv_source.items():
            if not isinstance(key, str) : continue
            kl = key.lower().replace(':','').strip() # Clean key for matching
            
            val_str = str(value) if not isinstance(value, dict) else (value.get("text") if isinstance(value.get("text"), str) else None)

            if "tender value" in kl and not data["key_financials"].get("tender_value_text"):
                data["key_financials"]["tender_value_text"] = val_str
                data["key_financials"]["tender_value_numeric"] = parse_amount_to_numeric(val_str)
            elif ("emd amount" in kl or "e.m.d amount" in kl) and not data["key_financials"].get("emd_amount_text"):
                data["key_financials"]["emd_amount_text"] = val_str
                data["key_financials"]["emd_amount_numeric"] = parse_amount_to_numeric(val_str)

            date_mapping_post = {
                "published date": "published_date", "publish date": "published_date",
                "document download / sale start date": "doc_download_start_date",
                "clarification start date": "clarification_start_date",
                "bid submission start date": "bid_submission_start_date",
                "bid submission end date": "closing_date", 
                "document download / sale end date": "doc_download_end_date",
                "clarification end date": "clarification_end_date",
                "bid opening date": "opening_date", 
                "financial bid opening date": "financial_bid_opening_date_actual"
            }
            if kl in date_mapping_post and isinstance(val_str, str) and val_str.strip():
                json_key = date_mapping_post[kl]
                if json_key not in data["critical_dates"]: 
                    data["critical_dates"][json_key] = val_str
            
            if kl == "location" and isinstance(val_str, str) and val_str.strip() and not data["location_info"].get("raw_location_text"):
                data["location_info"]["raw_location_text"] = val_str
            if kl == "pincode" and isinstance(val_str, str) and val_str.strip() and not data["location_info"].get("pincode"):
                 data["location_info"]["pincode"] = val_str
    
    title_for_cat = data.get("main_tender_info", {}).get("Tender Title") or \
                    data.get("main_tender_info", {}).get("Work Description")
    if not title_for_cat:
        for section_name, section_data in data.get("other_sections_kv", {}).items():
            if "work item details" in section_name.lower() or "work/item(s)" in section_name.lower():
                if isinstance(section_data, dict):
                    title_for_cat = section_data.get("Title") or section_data.get("Work Description") or section_data.get("Description")
                    if title_for_cat: break
    data["main_tender_info"]["Work Category (Raw Text)"] = title_for_cat if title_for_cat else None
    data["main_tender_info"]["Work Category (Classified)"] = "NEEDS_CLASSIFICATION"

    logger.info(f"Finished parsing {original_filename}. Main keys: {len(data['main_tender_info'])}, KV Sec: {len(data['other_sections_kv'])}, Table Sec: {len(data['other_sections_table'])}")
    return data

# --- Main CLI execution part ---
def main():
    global logger 
    parser = argparse.ArgumentParser(description="Parse ROT HTML summary file to structured JSON.")
    parser.add_argument("html_file_path", type=Path, help="Path to the ROT HTML summary file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Base directory to save JSON output. Site-key subdirs will be created.")
    parser.add_argument("--site-key", type=str, default=None, help="Site key (e.g., state name). Used for subfolder and data field.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (DEBUG level).")
    args = parser.parse_args()
    
    if args.verbose: logger.setLevel(logging.DEBUG)
    else: logger.setLevel(logging.INFO)

    html_file = args.html_file_path.resolve()
    if not html_file.is_file(): logger.error(f"Input HTML file not found: {html_file}"); return
    try:
        with open(html_file, 'r', encoding='utf-8', errors='replace') as f: html_content = f.read()
    except Exception as e: logger.error(f"Could not read HTML file {html_file}: {e}"); return

    structured_output = parse_rot_html_to_structured_data(html_content, html_file.name)
    if structured_output:
        site_key_to_use = args.site_key
        if not site_key_to_use: 
            try:
                if html_file.parent and html_file.parent.parent and html_file.parent.parent.name.lower() == "detailhtmls":
                     site_key_to_use = html_file.parent.name
            except Exception: logger.debug(f"Could not infer site_key from path for {html_file}.")
        if site_key_to_use: structured_output["source_site_key"] = site_key_to_use
        else: logger.warning("Site key not provided/inferred. 'source_site_key' will be null."); site_key_to_use = "unknown_site"
        if args.output_dir:
            output_base_dir = args.output_dir.resolve()
            final_output_dir = output_base_dir / site_key_to_use 
            final_output_dir.mkdir(parents=True, exist_ok=True)
            output_filename = html_file.stem + "_structured.json"
            output_file_path = final_output_dir / output_filename
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_output, f, indent=2, ensure_ascii=False)
                logger.info(f"Structured JSON saved to: {output_file_path}")
            except Exception as e: logger.error(f"Could not write JSON to file {output_file_path}: {e}")
        else:
            print(json.dumps(structured_output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
