#!/usr/bin/env python3
# File: site_controller.py (Orchestrates REGULAR tender scraping & offers manual ROT consolidation)

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
    print("FATAL (SiteController): Could not import 'run_scrape_for_one_site' from scrape.py.")
    scrape_regular_available = False
except Exception as e:
    print(f"FATAL (SiteController): Error importing scrape.py: {e}")
    scrape_regular_available = False

# --- Path Constants ---
try:
    PROJECT_ROOT = Path(__file__).parent.resolve()
except NameError:
    PROJECT_ROOT = Path('.').resolve()

SITE_DATA_ROOT = PROJECT_ROOT / "site_data"
REG_DATA_DIR = SITE_DATA_ROOT / "REG"
REG_MERGED_SITE_SPECIFIC_DIR = REG_DATA_DIR / "MergedSiteSpecific"
REG_FINAL_GLOBAL_MERGED_DIR = REG_DATA_DIR / "FinalGlobalMerged"
ROT_DATA_DIR = SITE_DATA_ROOT / "ROT"
ROT_MERGED_SITE_SPECIFIC_DIR = ROT_DATA_DIR / "MergedSiteSpecific" # Input for ROT consolidation
ROT_FINAL_GLOBAL_MERGED_DIR = ROT_DATA_DIR / "FinalGlobalMerged"   # Output for ROT consolidation
LOGS_BASE_DIR = PROJECT_ROOT / "LOGS"
SETTINGS_FILE_PATH = PROJECT_ROOT / "settings.json"
# --- END Path Constants ---

# --- Logger Setup ---
controller_logger = logging.getLogger('SiteController')
controller_logger.setLevel(logging.INFO)
controller_logger.propagate = False
if not controller_logger.hasHandlers():
    ch_controller = logging.StreamHandler()
    ch_controller.setFormatter(logging.Formatter("[%(levelname)s - SiteController] %(message)s"))
    controller_logger.addHandler(ch_controller)
    LOGS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    controller_log_file_path = LOGS_BASE_DIR / "site_controller.log"
    try:
        cfh_controller = logging.FileHandler(controller_log_file_path, mode='a', encoding='utf-8')
        cfh_controller.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] (SiteController) %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        controller_logger.addHandler(cfh_controller)
    except Exception as e_log_setup:
        print(f"ERROR (SiteController): Failed to set up file handler: {e_log_setup}")
# --- END Logger Setup ---

def load_orchestrator_settings() -> Tuple[Optional[Dict], Optional[Dict]]:
    # ... (This function remains unchanged, uses new SETTINGS_FILE_PATH) ...
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
        if not enabled_sites_to_process_regular: controller_logger.info("No sites enabled for REGULAR scraping.")
        else: controller_logger.info(f"Loaded {len(enabled_sites_to_process_regular)} enabled sites for REGULAR scraping.")
        return global_settings_from_file, enabled_sites_to_process_regular
    except Exception as e_load:
        controller_logger.error(f"Error loading settings from {SETTINGS_FILE_PATH}: {e_load}", exc_info=True)
        return None, None


async def inject_source_site_key_into_merged_file(site_key: str, site_merged_file_path: Path, data_type: str = "regular"):
    # ... (This function remains unchanged) ...
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
        modified_blocks = []; last_end_index = 0; found_any_blocks = False

        for match in tender_block_pattern.finditer(original_content):
            found_any_blocks = True
            modified_blocks.append(original_content[last_end_index:match.start()])
            start_marker, inner_content, end_marker = match.groups()
            if f"<SourceSiteKey>{site_key}</SourceSiteKey>" not in inner_content and "<SourceSiteKey>" not in inner_content:
                modified_block_content = f"{start_marker}\n{source_site_tag_to_inject}\n{inner_content.lstrip()}{end_marker}"
            else:
                if f"<SourceSiteKey>{site_key}</SourceSiteKey>" not in inner_content:
                    controller_logger.warning(f"INJECT ({data_type.upper()}): Block in {site_merged_file_path.name} already has a SourceSiteKey, but not '{site_key}'. Retaining original.")
                modified_block_content = match.group(0)
            modified_blocks.append(modified_block_content)
            last_end_index = match.end()

        if not found_any_blocks:
            controller_logger.warning(f"INJECT ({data_type.upper()}): No '--- TENDER START ---' blocks found in {site_merged_file_path.name}. File not modified.")
            return
        modified_blocks.append(original_content[last_end_index:])
        final_output_content = "".join(modified_blocks).strip()
        if final_output_content: final_output_content += "\n\n"
        site_merged_file_path.write_text(final_output_content, encoding="utf-8")
        controller_logger.info(f"INJECT ({data_type.upper()}): Successfully processed SourceSiteKey for {site_merged_file_path.name}")
    except Exception as e:
        controller_logger.error(f"INJECT ({data_type.upper()}): Error processing {site_merged_file_path.name} for SourceSiteKey: {e}", exc_info=True)


async def global_merge_site_specific_files(
    input_site_specific_merged_dir: Path,
    output_final_global_dir: Path,
    data_type: str # "regular" or "rot"
) -> Tuple[int, Optional[Path]]:
    # ... (This function definition is now more generic and correct, as verified before) ...
    # (It uses input_site_specific_merged_dir, output_final_global_dir, and data_type
    #  to determine file_pattern and output_filename_prefix)
    log_prefix = f"GLOBAL MERGE ({data_type.upper()})"
    controller_logger.info(f"--- {log_prefix}: Starting Merge of All Site-Specific Files ---")

    if not input_site_specific_merged_dir.is_dir():
        controller_logger.warning(f"{log_prefix}: Input directory not found: {input_site_specific_merged_dir}.")
        return 0, None

    file_pattern = "Merged_ROT_*.txt" if data_type == "rot" else "Merged_*.txt"
    output_filename_prefix = "Final_ROT_Tender_List_" if data_type == "rot" else "Final_Tender_List_"

    site_merged_files = list(input_site_specific_merged_dir.glob(file_pattern))
    if not site_merged_files:
        controller_logger.info(f"{log_prefix}: No '{file_pattern}' files found in {input_site_specific_merged_dir} for global merge.")
        return 0, None

    controller_logger.info(f"{log_prefix}: Found {len(site_merged_files)} site-specific {data_type.upper()} files for consolidation.")
    global_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    output_final_global_dir.mkdir(parents=True, exist_ok=True)
    global_final_output_path = output_final_global_dir / f"{output_filename_prefix}{global_timestamp}.txt"

    total_globally_unique_tender_blocks = 0; seen_global_tender_hashes: Set[int] = set()
    try:
        with open(global_final_output_path, "w", encoding="utf-8") as global_outfile:
            for site_file_path in site_merged_files:
                controller_logger.info(f"{log_prefix}: Processing: {site_file_path.name}")
                try:
                    site_file_content = site_file_path.read_text(encoding="utf-8", errors="replace").strip()
                    if not site_file_content: controller_logger.debug(f"{log_prefix}: File {site_file_path.name} empty."); continue
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
                            global_outfile.write(full_block_text + "\n\n"); seen_global_tender_hashes.add(block_hash_global); total_globally_unique_tender_blocks += 1
                except Exception as e_proc: controller_logger.error(f"{log_prefix}: Error processing {site_file_path.name}: {e_proc}")
        controller_logger.info(f"{log_prefix}: ✅ Merged {total_globally_unique_tender_blocks} unique blocks to: {global_final_output_path}")
        return total_globally_unique_tender_blocks, global_final_output_path
    except Exception as e_global:
        controller_logger.error(f"{log_prefix}: Fatal error during merge: {e_global}", exc_info=True)
        return total_globally_unique_tender_blocks, None


async def run_regular_scrape_orchestration(global_scraper_settings: Dict, enabled_sites_dict_regular: Dict):
    # ... (This function remains unchanged from its last corrected version) ...
    # (It correctly uses REG_MERGED_SITE_SPECIFIC_DIR and passes correct paths to global_merge_site_specific_files)
    if not scrape_regular_available:
        controller_logger.warning("Regular scraper (scrape.py) unavailable. Skipping regular scrape.")
        return 0, 0, 0, None
    if not enabled_sites_dict_regular:
        controller_logger.info("No sites enabled for regular scraping. Skipping.")
        return 0, 0, 0, None
    controller_logger.info("\n===== Orchestrating REGULAR Tender Scrape =====")
    sites_processed_count = 0; sites_failed_count = 0
    processed_site_merged_files_for_injection: List[Tuple[str, Path]] = []
    for site_key, site_config_data in enabled_sites_dict_regular.items():
        controller_logger.info(f"\n>>> REGULAR: Orchestrating site: {site_key} <<<")
        try:
            await run_scrape_for_one_site(site_key, site_config_data, global_scraper_settings)
            today_filename_part = datetime.datetime.now().strftime("%Y-%m-%d")
            safe_site_key_for_file = re.sub(r'[^\w\-]+', '_', site_key)
            expected_merged_file = REG_MERGED_SITE_SPECIFIC_DIR / f"Merged_{safe_site_key_for_file}_{today_filename_part}.txt"
            if expected_merged_file.is_file():
                processed_site_merged_files_for_injection.append((site_key, expected_merged_file))
            else:
                controller_logger.warning(f"REGULAR: Site-specific merged file for {site_key} not found at {expected_merged_file} after scrape run.")
            sites_processed_count += 1
        except Exception as e_orchestrate_site:
            controller_logger.error(f"REGULAR: XXX Orchestration for site {site_key} failed: {e_orchestrate_site} XXX", exc_info=True)
            sites_failed_count +=1
        controller_logger.info(f">>> REGULAR: Finished orchestration for {site_key} <<<")
    if processed_site_merged_files_for_injection:
        controller_logger.info("\n--- REGULAR: Injecting SourceSiteKey into Site-Specific Merged Files ---")
        for site_key_inject, merged_file_path_inject in processed_site_merged_files_for_injection:
            await inject_source_site_key_into_merged_file(site_key_inject, merged_file_path_inject, data_type="regular")
    else: controller_logger.info("REGULAR: No site-specific merged files found to inject SourceSiteKey.")
    controller_logger.info("\n===== REGULAR: Global Merge Phase =====")
    globally_merged_count_reg, global_final_file_reg = await global_merge_site_specific_files(
        REG_MERGED_SITE_SPECIFIC_DIR, REG_FINAL_GLOBAL_MERGED_DIR, data_type="regular"
    )
    successful_sites = sites_processed_count - sites_failed_count
    return successful_sites, sites_failed_count, globally_merged_count_reg, global_final_file_reg


async def main_controller(scrape_type_arg: str):
    controller_logger.info("===== Site Controller Initializing =====")
    controller_logger.info(f"Operation Type Requested: {scrape_type_arg.upper()}")
    overall_start_time = datetime.datetime.now()

    global_scraper_settings, enabled_sites_regular = load_orchestrator_settings()

    if global_scraper_settings is None: # load_orchestrator_settings returns None on critical failure
        controller_logger.critical("Exiting: Critical error loading settings for site_controller.")
        return

    # --- Regular Tender Scraping and Consolidation ---
    if scrape_type_arg in ["all", "regular"]:
        if not scrape_regular_available:
            controller_logger.error("Regular scraper (scrape.py) unavailable. Skipping regular scrape operations.")
        elif not enabled_sites_regular:
            controller_logger.info("No sites enabled for regular scraping. Skipping regular scrape operations.")
        else:
            s_succ_reg, s_fail_reg, g_merged_reg, g_file_reg = await run_regular_scrape_orchestration(global_scraper_settings, enabled_sites_regular)
            controller_logger.info(f"REGULAR SCRAPE & CONSOLIDATION SUMMARY: Sites Succeeded: {s_succ_reg}, Sites Failed: {s_fail_reg}, Globally Merged Tenders: {g_merged_reg}")
            if g_file_reg: controller_logger.info(f"REGULAR Global Output File: {g_file_reg}")
            else: controller_logger.warning("REGULAR Global Merge did not produce an output file.")
    
    # --- ROT Data Consolidation ---
    # This is now triggered if 'all' or 'rot_consolidate' is specified.
    # Individual ROT scrapes are manual via dashboard; this controller only consolidates.
    if scrape_type_arg in ["all", "rot_consolidate"]:
        controller_logger.info("\n===== Orchestrating ROT Data Consolidation (if site-specific files exist) =====")
        g_merged_rot, g_file_rot = await global_merge_site_specific_files(
            ROT_MERGED_SITE_SPECIFIC_DIR, # Input directory
            ROT_FINAL_GLOBAL_MERGED_DIR,  # Output directory
            data_type="rot"
        )
        controller_logger.info(f"ROT CONSOLIDATION SUMMARY: Globally Merged ROT Tenders: {g_merged_rot}")
        if g_file_rot: controller_logger.info(f"ROT Global Output File: {g_file_rot}")
        else: controller_logger.warning("ROT Global Consolidation did not produce an output file (this is expected if no site-specific Merged_ROT_*.txt files were found).")
    
    # Handles old "--type rot" calls gracefully
    elif scrape_type_arg == "rot" and scrape_type_arg not in ["all", "rot_consolidate"]:
         controller_logger.info("Received '--type rot'. Individual ROT scraping is manual via the dashboard. To consolidate existing ROT data, use '--type rot_consolidate' or '--type all'. No action taken for direct ROT scraping by controller.")

    overall_end_time = datetime.datetime.now()
    overall_duration = overall_end_time - overall_start_time
    controller_logger.info(f"\nTotal Site Controller Duration: {overall_duration}")
    controller_logger.info("===== Site Controller Finished =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TenFin Site Controller: Handles regular tender scraping and data consolidation for REG and ROT types.")
    parser.add_argument(
        "--type",
        type=str,
        choices=["regular", "rot_consolidate", "all"], # MODIFIED: "rot" direct scraping removed
        default="all",
        help=("Type of operation: "
              "'regular' to scrape regular tenders & consolidate them; "
              "'rot_consolidate' to consolidate existing site-specific ROT data; "
              "'all' to perform regular scrape & consolidation AND ROT data consolidation.")
    )
    args = parser.parse_args()

    # Playwright check (not strictly needed if only consolidating, but good for 'regular'/'all')
    if args.type in ["regular", "all"]:
        try:
            from playwright.sync_api import sync_playwright # Keep for this check
            with sync_playwright() as p_test:
                browser_test = p_test.chromium.launch(headless=True); browser_test.close()
            controller_logger.info("Playwright check OK for Site Controller (needed for regular scraping).")
        except Exception as play_err_main:
             controller_logger.critical(f"Playwright check failed: {play_err_main}. Regular scraping might fail. Install with 'python -m playwright install --with-deps'");
             if args.type != "rot_consolidate": # Only exit if playwright is actually needed
                 sys.exit(1)
    
    if args.type in ["regular", "all"] and not scrape_regular_available:
        controller_logger.error("Cannot run 'regular' or 'all' operations: scrape.py or its 'run_scrape_for_one_site' function is unavailable.");
        sys.exit(1)
    
    asyncio.run(main_controller(args.type))
