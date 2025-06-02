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
    from scrape import run_scrape_for_one_site # scrape.py will be modified later
    scrape_regular_available = True
except ImportError:
    print("FATAL: Could not import 'run_scrape_for_one_site' from scrape.py.")
    scrape_regular_available = False
except Exception as e:
    print(f"FATAL: Error importing scrape.py: {e}")
    scrape_regular_available = False

# --- ROT Scraper Import (Primarily for URL_SUFFIX_ROT if dashboard.py fails to get it from worker) ---
# This controller does not directly execute ROT scraping.
try:
    from headless_rot_worker import URL_SUFFIX_ROT
except ImportError:
    try:
        from scrape_rot import URL_SUFFIX_ROT # Older fallback
    except ImportError:
        print("WARNING (SiteController): Could not import URL_SUFFIX_ROT from headless_rot_worker or scrape_rot.")
        URL_SUFFIX_ROT = "?page=WebTenderStatusLists&service=page" # Default, should match worker/dashboard
    except Exception as e_rot_import_fallback:
        print(f"WARNING (SiteController): Error during ROT suffix fallback import: {e_rot_import_fallback}")
        URL_SUFFIX_ROT = "?page=WebTenderStatusLists&service=page"
except Exception as e_headless_import:
    print(f"WARNING (SiteController): Error importing from headless_rot_worker: {e_headless_import}")
    URL_SUFFIX_ROT = "?page=WebTenderStatusLists&service=page"


# --- NEW UNIFIED PATH CONFIGURATION ---
try:
    PROJECT_ROOT = Path(__file__).parent.resolve()
except NameError:
    PROJECT_ROOT = Path('.').resolve()

SITE_DATA_ROOT = PROJECT_ROOT / "site_data"

# REG Specific Paths (site_controller.py primarily deals with these for global merge)
REG_DATA_DIR = SITE_DATA_ROOT / "REG"
REG_MERGED_SITE_SPECIFIC_DIR = REG_DATA_DIR / "MergedSiteSpecific" # Input for this controller's global merge
REG_FINAL_GLOBAL_MERGED_DIR = REG_DATA_DIR / "FinalGlobalMerged"   # Output of this controller's global merge

# LOGS Path (Unified)
LOGS_BASE_DIR = PROJECT_ROOT / "LOGS"

# Settings file path
SETTINGS_FILE_PATH = PROJECT_ROOT / "settings.json"
# --- END NEW UNIFIED PATH CONFIGURATION ---


# --- SiteController Logger Setup ---
controller_logger = logging.getLogger('SiteController')
controller_logger.setLevel(logging.INFO)
controller_logger.propagate = False
if not controller_logger.hasHandlers():
    # Console Handler
    ch_controller = logging.StreamHandler()
    ch_controller.setFormatter(logging.Formatter("[%(levelname)s - SiteController] %(message)s"))
    controller_logger.addHandler(ch_controller)

    # File Handler - MODIFIED to use new LOGS_BASE_DIR
    LOGS_BASE_DIR.mkdir(parents=True, exist_ok=True) # Ensure base log dir exists
    controller_log_file_path = LOGS_BASE_DIR / "site_controller.log"
    try:
        cfh_controller = logging.FileHandler(controller_log_file_path, mode='a', encoding='utf-8')
        cfh_controller.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] (SiteController) %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        controller_logger.addHandler(cfh_controller)
    except Exception as e_log_setup:
        # Use print here as logger might not be fully set up for file if this fails
        print(f"ERROR (SiteController): Failed to set up file handler for site_controller.log: {e_log_setup}")
# --- END SiteController Logger Setup ---


def load_orchestrator_settings() -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Loads global scraper settings and site configurations for REGULAR tenders.
    Uses the globally defined SETTINGS_FILE_PATH.
    """
    if not SETTINGS_FILE_PATH.is_file():
        controller_logger.error(f"FATAL: Settings file not found at {SETTINGS_FILE_PATH}")
        return None, None
    try:
        with open(SETTINGS_FILE_PATH, 'r', encoding='utf-8') as f:
            settings_data = json.load(f)

        global_settings_from_file = settings_data.get("global_scraper_settings", {})
        all_site_configs = settings_data.get("site_configurations", {})

        if not all_site_configs:
            controller_logger.warning("No 'site_configurations' found in settings.json.")
            return global_settings_from_file, {} # Return empty dict for sites

        enabled_sites_to_process_regular = {
            site_name: config_data
            for site_name, config_data in all_site_configs.items()
            if config_data.get("enabled", False) # Check if site is enabled
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
        controller_logger.error(f"An unexpected error occurred while loading settings from {SETTINGS_FILE_PATH}: {e_load}", exc_info=True)
        return None, None


async def inject_source_site_key_into_merged_file(site_key: str, site_merged_file_path: Path, data_type: str = "regular"):
    """
    Injects <SourceSiteKey> into each tender block of a site-specific merged file.
    This function assumes it's only called for "regular" data type by this controller.
    The site_merged_file_path will be within REG_MERGED_SITE_SPECIFIC_DIR.
    """
    if not site_merged_file_path.is_file():
        controller_logger.warning(f"INJECT ({data_type.upper()}): Site merged file not found: {site_merged_file_path}")
        return
    controller_logger.info(f"INJECT ({data_type.upper()}): Injecting SourceSiteKey '{site_key}' into {site_merged_file_path.name}")
    try:
        original_content = site_merged_file_path.read_text(encoding="utf-8")
        if not original_content.strip():
            controller_logger.info(f"INJECT ({data_type.upper()}): File {site_merged_file_path.name} is empty. Skipping.")
            return

        tender_block_pattern = re.compile(r"(--- TENDER START ---)([\s\S]*?)(--- TENDER END ---)")
        source_site_tag_to_inject = f"<SourceSiteKey>{site_key}</SourceSiteKey>"
        
        modified_blocks = []
        last_end_index = 0
        found_any_blocks = False

        for match in tender_block_pattern.finditer(original_content):
            found_any_blocks = True
            # Append content before the current match
            modified_blocks.append(original_content[last_end_index:match.start()])
            
            start_marker, inner_content, end_marker = match.groups()
            
            # Ensure SourceSiteKey is not already present (idempotency)
            if f"<SourceSiteKey>{site_key}</SourceSiteKey>" not in inner_content and \
               "<SourceSiteKey>" not in inner_content: # More general check if key exists with different value
                # Add the new tag, preferrably after --- TENDER START ---
                # and before other content.
                # lstrip to remove leading newlines from original inner_content if any.
                modified_block_content = f"{start_marker}\n{source_site_tag_to_inject}\n{inner_content.lstrip()}{end_marker}"
            else: # Already contains a SourceSiteKey or the exact one
                if f"<SourceSiteKey>{site_key}</SourceSiteKey>" not in inner_content:
                    controller_logger.warning(f"INJECT ({data_type.upper()}): Block in {site_merged_file_path.name} already has a SourceSiteKey, but not '{site_key}'. Retaining original.")
                modified_block_content = match.group(0) # Use the whole original matched block

            modified_blocks.append(modified_block_content)
            last_end_index = match.end()

        if not found_any_blocks:
            controller_logger.warning(f"INJECT ({data_type.upper()}): No '--- TENDER START ---' blocks found in {site_merged_file_path.name}. File not modified.")
            return

        # Append any remaining content after the last match
        modified_blocks.append(original_content[last_end_index:])
        
        final_output_content = "".join(modified_blocks).strip()
        if final_output_content: # Add trailing newlines only if there's content
            final_output_content += "\n\n" 
            
        site_merged_file_path.write_text(final_output_content, encoding="utf-8")
        controller_logger.info(f"INJECT ({data_type.upper()}): Successfully processed SourceSiteKey for {site_merged_file_path.name}")
    except Exception as e:
        controller_logger.error(f"INJECT ({data_type.upper()}): Error processing {site_merged_file_path.name} for SourceSiteKey: {e}", exc_info=True)

async def global_merge_site_specific_files(
    # site_specific_merged_dir argument will now be REG_MERGED_SITE_SPECIFIC_DIR
    site_specific_merged_dir: Path,
    # final_global_output_dir argument will now be REG_FINAL_GLOBAL_MERGED_DIR
    final_global_output_dir: Path,
    data_type: str = "regular" # This controller only handles "regular"
) -> Tuple[int, Optional[Path]]:
    log_prefix = f"GLOBAL MERGE ({data_type.upper()})" # Will always be REGULAR from this script
    controller_logger.info(f"--- {log_prefix}: Starting Merge of All Site-Specific Files ---")

    if not site_specific_merged_dir.is_dir():
        controller_logger.warning(f"{log_prefix}: Input directory for site-specific merged files not found: {site_specific_merged_dir}. No global merge will occur.")
        return 0, None

    file_pattern = "Merged_*.txt" # Pattern for regular site-specific merged files
    if data_type != "regular": # This controller no longer handles data_type == "rot"
        controller_logger.error(f"{log_prefix}: This function in site_controller.py is intended only for 'regular' data type. Received: {data_type}. Aborting merge.");
        return 0, None

    site_merged_files = list(site_specific_merged_dir.glob(file_pattern))
    if not site_merged_files:
        controller_logger.info(f"{log_prefix}: No '{file_pattern}' files found in {site_specific_merged_dir} for global merge.")
        return 0, None

    controller_logger.info(f"{log_prefix}: Found {len(site_merged_files)} site-specific REGULAR files for consolidation.")
    global_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_filename_prefix = "Final_Tender_List_" # For regular tenders

    final_global_output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
    global_final_output_path = final_global_output_dir / f"{output_filename_prefix}{global_timestamp}.txt"

    total_globally_unique_tender_blocks = 0
    seen_global_tender_hashes: Set[int] = set()
    try:
        with open(global_final_output_path, "w", encoding="utf-8") as global_outfile:
            for site_file_path in site_merged_files:
                controller_logger.info(f"{log_prefix}: Processing: {site_file_path.name}")
                try:
                    site_file_content = site_file_path.read_text(encoding="utf-8", errors="replace").strip()
                    if not site_file_content:
                        controller_logger.debug(f"{log_prefix}: File {site_file_path.name} is empty. Skipping.")
                        continue

                    raw_blocks_with_delimiter = site_file_content.split("--- TENDER END ---")
                    tender_blocks_from_site_file = []
                    for block_segment in raw_blocks_with_delimiter:
                        if "--- TENDER START ---" in block_segment:
                            start_index = block_segment.find("--- TENDER START ---")
                            if start_index != -1:
                                reconstructed_block = block_segment[start_index:].strip() + "\n--- TENDER END ---"
                                tender_blocks_from_site_file.append(reconstructed_block.strip())

                    for full_block_text in tender_blocks_from_site_file:
                        if not full_block_text: continue
                        block_hash_global = hash(full_block_text)
                        if block_hash_global not in seen_global_tender_hashes:
                            global_outfile.write(full_block_text + "\n\n")
                            seen_global_tender_hashes.add(block_hash_global)
                            total_globally_unique_tender_blocks += 1
                except Exception as e_proc_site_file:
                    controller_logger.error(f"{log_prefix}: Error processing site-specific file {site_file_path.name}: {e_proc_site_file}")
        
        controller_logger.info(f"{log_prefix}: ✅ Merged {total_globally_unique_tender_blocks} unique REGULAR tender blocks to: {global_final_output_path}")
        return total_globally_unique_tender_blocks, global_final_output_path
    except Exception as e_global_merge:
        controller_logger.error(f"{log_prefix}: Fatal error during global merge of REGULAR tenders: {e_global_merge}", exc_info=True)
        return total_globally_unique_tender_blocks, None

async def run_regular_scrape_orchestration(global_scraper_settings: Dict, enabled_sites_dict_regular: Dict):
    if not scrape_regular_available:
        controller_logger.warning("Regular scraper (scrape.py) unavailable. Skipping regular scrape.")
        return 0, 0, 0, None # Succeeded, Failed, MergedCount, MergedPath
    if not enabled_sites_dict_regular:
        controller_logger.info("No sites enabled for regular scraping. Skipping.")
        return 0, 0, 0, None

    controller_logger.info("\n===== Orchestrating REGULAR Tender Scrape =====")
    sites_processed_count = 0
    sites_failed_count = 0
    processed_site_merged_files_for_injection: List[Tuple[str, Path]] = []

    for site_key, site_config_data in enabled_sites_dict_regular.items():
        controller_logger.info(f"\n>>> REGULAR: Orchestrating site: {site_key} <<<")
        try:
            # scrape.py's run_scrape_for_one_site will now use new paths internally
            # and save its Merged_{SITE_KEY}_{DATE}.txt to REG_MERGED_SITE_SPECIFIC_DIR
            await run_scrape_for_one_site(site_key, site_config_data, global_scraper_settings)

            today_filename_part = datetime.datetime.now().strftime("%Y-%m-%d")
            safe_site_key_for_file = re.sub(r'[^\w\-]+', '_', site_key)
            
            # Check for the expected merged file in the new location
            expected_merged_file = REG_MERGED_SITE_SPECIFIC_DIR / f"Merged_{safe_site_key_for_file}_{today_filename_part}.txt"
            
            if expected_merged_file.is_file():
                processed_site_merged_files_for_injection.append((site_key, expected_merged_file))
            else:
                controller_logger.warning(f"REGULAR: Site-specific merged file for {site_key} not found at {expected_merged_file} after scrape run. This might be okay if no tenders were found for this site.")
            sites_processed_count += 1 # Count as processed even if no file, scrape.py ran.
        except Exception as e_orchestrate_site:
            controller_logger.error(f"REGULAR: XXX Orchestration for site {site_key} failed: {e_orchestrate_site} XXX", exc_info=True)
            sites_failed_count +=1
        controller_logger.info(f">>> REGULAR: Finished orchestration for {site_key} <<<")

    if processed_site_merged_files_for_injection:
        controller_logger.info("\n--- REGULAR: Injecting SourceSiteKey into Site-Specific Merged Files ---")
        for site_key_inject, merged_file_path_inject in processed_site_merged_files_for_injection:
            # inject_source_site_key_into_merged_file operates on the path given
            await inject_source_site_key_into_merged_file(site_key_inject, merged_file_path_inject, data_type="regular")
    else:
        controller_logger.info("REGULAR: No site-specific merged files found to inject SourceSiteKey (this is normal if no tenders were found for any enabled site).")

    controller_logger.info("\n===== REGULAR: Global Merge Phase =====")
    # Pass new paths to global_merge_site_specific_files
    globally_merged_count_reg, global_final_file_reg = await global_merge_site_specific_files(
        REG_MERGED_SITE_SPECIFIC_DIR, # Input directory (e.g., site_data/REG/MergedSiteSpecific)
        REG_FINAL_GLOBAL_MERGED_DIR,  # Output directory (e.g., site_data/REG/FinalGlobalMerged)
        data_type="regular"
    )
    successful_sites = sites_processed_count - sites_failed_count
    return successful_sites, sites_failed_count, globally_merged_count_reg, global_final_file_reg

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
