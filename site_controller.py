#!/usr/bin/env python3
# File: site_controller.py (Orchestrates scraping for REGULAR tenders)

import asyncio
import json
from pathlib import Path
import sys
import logging
import datetime
import re
from typing import Tuple, Optional, Dict, Set, List, Any
import argparse 

# --- Import Regular Scraper ---
try:
    from scrape import run_scrape_for_one_site
    scrape_regular_available = True
except ImportError:
    print("FATAL: Could not import 'run_scrape_for_one_site' from scrape.py.")
    scrape_regular_available = False
except Exception as e:
    print(f"FATAL: Error importing scrape.py: {e}")
    scrape_regular_available = False

# --- ROT Scraper Import (for URL_SUFFIX_ROT, not for direct execution by controller anymore) ---
try:
    from scrape_rot import URL_SUFFIX_ROT # Keep for settings derivation if needed
    # scrape_rot_available = True # Not strictly needed by controller anymore for execution
except ImportError:
    print("WARNING: Could not import URL_SUFFIX_ROT from scrape_rot.py. ROT settings derivation might be affected if used.")
    URL_SUFFIX_ROT = "?page=WebTenderStatusLists&service=page" # Fallback
except Exception as e:
    print(f"WARNING: Error importing from scrape_rot.py: {e}.")
    URL_SUFFIX_ROT = "?page=WebTenderStatusLists&service=page"


try:
    SCRIPT_DIR_CONTROLLER = Path(__file__).parent.resolve()
except NameError:
    SCRIPT_DIR_CONTROLLER = Path('.').resolve()

SETTINGS_FILE_PATH = SCRIPT_DIR_CONTROLLER / "settings.json"
BASE_DATA_DIR_CONTROLLER_ROOT = SCRIPT_DIR_CONTROLLER / "scraped_data"
SITE_SPECIFIC_MERGED_DIR = BASE_DATA_DIR_CONTROLLER_ROOT / "SiteSpecificMerged"

# ROT paths are managed by scrape_rot.py and dashboard.py directly now
# BASE_DATA_DIR_ROT_CONTROLLER_ROOT = SCRIPT_DIR_CONTROLLER / "scraped_data_rot"
# SITE_SPECIFIC_MERGED_DIR_ROT = BASE_DATA_DIR_ROT_CONTROLLER_ROOT / "SiteSpecificMergedROT"


controller_logger = logging.getLogger('SiteController')
controller_logger.setLevel(logging.INFO)
controller_logger.propagate = False
if not controller_logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s - SiteController] %(message)s"))
    controller_logger.addHandler(ch)
    controller_log_file = SCRIPT_DIR_CONTROLLER / "logs" / "site_controller.log"
    controller_log_file.parent.mkdir(parents=True, exist_ok=True)
    cfh = logging.FileHandler(controller_log_file, mode='a', encoding='utf-8')
    cfh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] (SiteController) %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    controller_logger.addHandler(cfh)


def load_orchestrator_settings() -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Loads global scraper settings and site configurations for REGULAR tenders.
    Returns: (global_settings, enabled_sites_regular)
    """
    if not SETTINGS_FILE_PATH.is_file():
        controller_logger.error(f"FATAL: Settings file not found at {SETTINGS_FILE_PATH}")
        return None, None
    try:
        with open(SETTINGS_FILE_PATH, 'r', encoding='utf-8') as f: settings_data = json.load(f)
        
        global_settings_from_file = settings_data.get("global_scraper_settings", {})
        all_site_configs = settings_data.get("site_configurations", {})

        if not all_site_configs:
            controller_logger.warning("No 'site_configurations' found in settings.json.")
            return global_settings_from_file, {}

        enabled_sites_to_process_regular = {
            site_name: config_data 
            for site_name, config_data in all_site_configs.items() 
            if config_data.get("enabled", False)
        }
        if not enabled_sites_to_process_regular:
            controller_logger.info("No sites are currently enabled for REGULAR scraping in settings.json.")
        else:
            controller_logger.info(f"Loaded configurations for {len(enabled_sites_to_process_regular)} enabled sites for REGULAR scraping.")

        return global_settings_from_file, enabled_sites_to_process_regular

    except json.JSONDecodeError as e_json:
        controller_logger.error(f"Error decoding JSON from {SETTINGS_FILE_PATH}: {e_json}")
        return None, None
    except Exception as e_load:
        controller_logger.error(f"An unexpected error occurred while loading settings: {e_load}", exc_info=True)
        return None, None


async def inject_source_site_key_into_merged_file(site_key: str, site_merged_file_path: Path, data_type: str = "regular"):
    # This function remains largely the same, data_type helps with logging.
    # It will only be called with data_type="regular" by this controller now.
    if not site_merged_file_path.is_file():
        controller_logger.warning(f"INJECT ({data_type.upper()}): Site merged file not found: {site_merged_file_path}")
        return
    controller_logger.info(f"INJECT ({data_type.upper()}): Injecting SourceSiteKey '{site_key}' into {site_merged_file_path.name}")
    try:
        original_content = site_merged_file_path.read_text(encoding="utf-8")
        if not original_content.strip():
            controller_logger.info(f"INJECT ({data_type.upper()}): File {site_merged_file_path.name} empty. Skipping."); return
        
        tender_block_pattern = re.compile(r"(--- TENDER START ---)([\s\S]*?)(--- TENDER END ---)")
        source_site_tag_to_inject = f"<SourceSiteKey>{site_key}</SourceSiteKey>"
        processed_blocks = []; last_end = 0; match_found_in_file = False
        for match in tender_block_pattern.finditer(original_content):
            match_found_in_file = True
            start_marker, inner_content, end_marker = match.groups()
            processed_blocks.append(original_content[last_end:match.start()])
            stripped_inner_content = inner_content.lstrip('\r\n') 
            modified_block_content = f"{start_marker}\n{source_site_tag_to_inject}\n{stripped_inner_content.strip()}\n{end_marker}"
            processed_blocks.append(modified_block_content)
            last_end = match.end()
        if not match_found_in_file:
            controller_logger.warning(f"INJECT ({data_type.upper()}): No tender blocks in {site_merged_file_path.name}. Not modified."); return 
        processed_blocks.append(original_content[last_end:])
        modified_content = "".join(processed_blocks)
        if source_site_tag_to_inject not in modified_content and match_found_in_file:
            controller_logger.error(f"INJECT ({data_type.upper()}): CRITICAL! Tag NOT found post-processing {site_merged_file_path.name}.")
        final_output_content = modified_content.strip()
        if final_output_content: final_output_content += "\n\n"
        site_merged_file_path.write_text(final_output_content, encoding="utf-8")
        controller_logger.info(f"INJECT ({data_type.upper()}): Successfully wrote to {site_merged_file_path.name}")
    except Exception as e:
        controller_logger.error(f"INJECT ({data_type.upper()}): Error for {site_merged_file_path.name}: {e}", exc_info=True)


async def global_merge_site_specific_files(
    site_specific_merged_dir: Path,
    final_global_output_dir: Path,
    data_type: str = "regular" 
) -> Tuple[int, Optional[Path]]:
    # This function remains the same, but will only be called with data_type="regular" by this controller.
    log_prefix = f"GLOBAL MERGE ({data_type.upper()})"
    controller_logger.info(f"--- {log_prefix}: Starting Merge of All Site-Specific Files ---")
    if not site_specific_merged_dir.is_dir():
        controller_logger.warning(f"{log_prefix}: Dir not found: {site_specific_merged_dir}. No global merge."); return 0, None
    file_pattern = "Merged_*.txt" # Only regular merge files now
    if data_type == "rot": # Should not happen from this controller anymore
        controller_logger.error(f"{log_prefix}: This controller should not be merging ROT files. Check logic."); return 0,None
    site_merged_files = list(site_specific_merged_dir.glob(file_pattern))
    if not site_merged_files:
        controller_logger.info(f"{log_prefix}: No '{file_pattern}' files found for global merge."); return 0, None
    controller_logger.info(f"{log_prefix}: Found {len(site_merged_files)} site-specific files for consolidation.")
    global_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_filename_prefix = "Final_Tender_List_" # Only regular
    global_final_output_path = final_global_output_dir / f"{output_filename_prefix}{global_timestamp}.txt"
    final_global_output_dir.mkdir(parents=True, exist_ok=True)
    total_globally_unique_tender_blocks = 0; seen_global_tender_hashes: Set[int] = set()
    try:
        with open(global_final_output_path, "w", encoding="utf-8") as global_outfile:
            for site_file_path in site_merged_files:
                controller_logger.info(f"{log_prefix}: Processing: {site_file_path.name}")
                try:
                    site_file_content = site_file_path.read_text(encoding="utf-8", errors="replace").strip()
                    if not site_file_content: controller_logger.debug(f"{log_prefix}: File {site_file_path.name} empty. Skipping."); continue
                    raw_blocks = site_file_content.split("--- TENDER END ---")
                    tender_blocks_from_site_file = [rb.strip() + "\n--- TENDER END ---" for rb in raw_blocks if "--- TENDER START ---" in rb.strip()]
                    for full_block_text in tender_blocks_from_site_file:
                        clean_block = full_block_text.strip() 
                        if not clean_block: continue
                        block_hash_global = hash(clean_block)
                        if block_hash_global not in seen_global_tender_hashes:
                            global_outfile.write(clean_block + "\n\n"); seen_global_tender_hashes.add(block_hash_global); total_globally_unique_tender_blocks += 1
                except Exception as e_proc_site_file: controller_logger.error(f"{log_prefix}: Error processing {site_file_path.name}: {e_proc_site_file}")
        controller_logger.info(f"{log_prefix}: ✅ Merged {total_globally_unique_tender_blocks} unique blocks to: {global_final_output_path}"); return total_globally_unique_tender_blocks, global_final_output_path
    except Exception as e_global_merge:
        controller_logger.error(f"{log_prefix}: Fatal error: {e_global_merge}", exc_info=True); return total_globally_unique_tender_blocks, None


async def run_regular_scrape_orchestration(global_scraper_settings: Dict, enabled_sites_dict_regular: Dict):
    if not scrape_regular_available:
        controller_logger.warning("Regular scraper (scrape.py) unavailable. Skipping regular scrape.")
        return 0, 0, 0, None # Ensure consistent return tuple
    if not enabled_sites_dict_regular:
        controller_logger.info("No sites enabled for regular scraping. Skipping."); return 0,0,0,None

    controller_logger.info("\n===== Orchestrating REGULAR Tender Scrape =====")
    sites_processed_count = 0; sites_failed_count = 0
    processed_site_merged_files_for_injection = []
    for site_key, site_config_data in enabled_sites_dict_regular.items():
        controller_logger.info(f"\n>>> REGULAR: Orchestrating site: {site_key} <<<")
        try:
            await run_scrape_for_one_site(site_key, site_config_data, global_scraper_settings)
            today_filename_part = datetime.datetime.now().strftime("%Y-%m-%d")
            safe_site_key_for_file = re.sub(r'[^\w\-]+', '_', site_key)
            expected_merged_file = SITE_SPECIFIC_MERGED_DIR / f"Merged_{safe_site_key_for_file}_{today_filename_part}.txt"
            if expected_merged_file.is_file(): processed_site_merged_files_for_injection.append((site_key, expected_merged_file))
            else: controller_logger.warning(f"REGULAR: Merged file for {site_key} not found at {expected_merged_file}.")
            sites_processed_count += 1
        except Exception as e_orchestrate_site:
            controller_logger.error(f"REGULAR: XXX Orchestration for {site_key} failed: {e_orchestrate_site} XXX", exc_info=True)
            sites_failed_count +=1
        controller_logger.info(f">>> REGULAR: Finished orchestration for {site_key} <<<")

    if processed_site_merged_files_for_injection:
        controller_logger.info("\n--- REGULAR: Injecting SourceSiteKey ---")
        for site_key, merged_file_path in processed_site_merged_files_for_injection:
            await inject_source_site_key_into_merged_file(site_key, merged_file_path, data_type="regular")
    else: controller_logger.info("REGULAR: No site-specific files to inject SourceSiteKey.")
    controller_logger.info("\n===== REGULAR: Global Merge Phase =====")
    globally_merged_count_reg, global_final_file_reg = await global_merge_site_specific_files(
        SITE_SPECIFIC_MERGED_DIR, BASE_DATA_DIR_CONTROLLER_ROOT, data_type="regular"
    )
    return sites_processed_count - sites_failed_count, sites_failed_count, globally_merged_count_reg, global_final_file_reg


# run_rot_scrape_orchestration function is REMOVED as ROT is now manual per site via dashboard


async def main_controller(scrape_type_arg: str):
    controller_logger.info("===== Site Controller Initializing =====")
    controller_logger.info(f"Scrape Type Requested by cron/manual call: {scrape_type_arg.upper()}")
    overall_start_time = datetime.datetime.now()
    
    global_scraper_settings, enabled_sites_regular = load_orchestrator_settings()
    
    if global_scraper_settings is None: 
        controller_logger.critical("Exiting: Critical error loading settings."); return

    if scrape_type_arg == "rot":
        controller_logger.info("ROT scraping is now handled manually via the dashboard for interactive CAPTCHA.")
        controller_logger.info("This controller run (if triggered for ROT) will not perform bulk ROT scraping.")
        # No further action for ROT here.
    
    # Regular Scrape (handles "regular" and "all" effectively becoming just "regular")
    if scrape_type_arg in ["all", "regular"]:
        if not scrape_regular_available:
            controller_logger.error("Regular scraper (scrape.py) unavailable. Skipping.")
        elif not enabled_sites_regular:
            controller_logger.info("No sites enabled for regular scraping. Skipping.")
        else:
            s_succ_reg, s_fail_reg, g_merged_reg, g_file_reg = await run_regular_scrape_orchestration(global_scraper_settings, enabled_sites_regular)
            controller_logger.info(f"REGULAR SCRAPE SUMMARY: Sites Succeeded: {s_succ_reg}, Sites Failed: {s_fail_reg}, Globally Merged Tenders: {g_merged_reg}")
            if g_file_reg: controller_logger.info(f"REGULAR Global Output: {g_file_reg}")
    
    overall_end_time = datetime.datetime.now()
    overall_duration = overall_end_time - overall_start_time
    controller_logger.info(f"\nTotal Site Controller Duration: {overall_duration}")
    controller_logger.info("===== Site Controller Finished =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TenFin Site Controller")
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["regular", "all", "rot"], # Keep "rot" choice to catch old cron jobs gracefully
        default="all", 
        help="Type of scrape to perform: 'regular', 'all' (effectively regular), or 'rot' (now logs manual info)."
    )
    args = parser.parse_args()

    try:
        import playwright.sync_api
        with playwright.sync_api.sync_playwright() as p_test:
            browser_test = p_test.chromium.launch(headless=True); browser_test.close()
        controller_logger.info("Playwright check OK for Site Controller.")
    except Exception as play_err_main:
         controller_logger.critical(f"Playwright check failed: {play_err_main}. Install with 'python -m playwright install --with-deps'"); sys.exit(1)
    
    if args.type in ["regular", "all"] and not scrape_regular_available:
        controller_logger.error("Cannot run 'regular' scrape: scrape.py unavailable."); sys.exit(1)
    
    asyncio.run(main_controller(args.type))
