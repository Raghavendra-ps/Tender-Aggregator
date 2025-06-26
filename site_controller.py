#!/usr/bin/env python3
# File: site_controller.py (Orchestrates REGULAR tender scraping)

import asyncio
import json
from pathlib import Path
import sys
import logging
import datetime
import argparse
from typing import Dict

# --- Import Regular Scraper ---
try:
    from scrape import run_scrape_for_one_site
    scrape_regular_available = True
except ImportError:
    print("FATAL (SiteController): Could not import 'run_scrape_for_one_site' from scrape.py.")
    sys.exit(1)
except Exception as e:
    print(f"FATAL (SiteController): Error importing scrape.py: {e}")
    sys.exit(1)

# --- Path Constants ---
PROJECT_ROOT = Path(__file__).parent.resolve()
LOGS_BASE_DIR = PROJECT_ROOT / "LOGS"
SETTINGS_FILE_PATH = PROJECT_ROOT / "settings.json"
# --- NEW: Lock file for status checking ---
SCRAPE_LOCK_FILE = PROJECT_ROOT / "scrape_in_progress.lock"

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

def load_orchestrator_settings():
    if not SETTINGS_FILE_PATH.is_file():
        controller_logger.error(f"FATAL: Settings file not found at {SETTINGS_FILE_PATH}")
        return None, None
    try:
        with open(SETTINGS_FILE_PATH, 'r', encoding='utf-8') as f:
            settings_data = json.load(f)
        global_settings = settings_data.get("global_scraper_settings", {})
        all_site_configs = settings_data.get("site_configurations", {})
        
        enabled_sites = {
            site_name: config
            for site_name, config in all_site_configs.items()
            if config.get("enabled", False)
        }
        
        if not enabled_sites:
            controller_logger.info("No sites enabled for scraping.")
        else:
            controller_logger.info(f"Loaded {len(enabled_sites)} enabled sites for scraping.")
            
        return global_settings, enabled_sites
    except Exception as e:
        controller_logger.error(f"Error loading settings from {SETTINGS_FILE_PATH}: {e}", exc_info=True)
        return None, None

async def run_regular_scrape_orchestration(global_scraper_settings: Dict, enabled_sites_dict: Dict):
    if not scrape_regular_available:
        controller_logger.warning("Regular scraper (scrape.py) unavailable. Skipping regular scrape.")
        return 0, 0
    if not enabled_sites_dict:
        controller_logger.info("No sites enabled for regular scraping. Skipping.")
        return 0, 0

    controller_logger.info("\n===== Orchestrating REGULAR Tender Scrape into Database =====")
    controller_logger.info(f"Sites to be processed: {list(enabled_sites_dict.keys())}")
    
    sites_processed_count = 0
    sites_failed_count = 0
    
    for site_key, site_config_data in enabled_sites_dict.items():
        try:
            controller_logger.info(f"\n>>> Starting scrape for site: {site_key} <<<")
            await run_scrape_for_one_site(site_key, site_config_data, global_scraper_settings)
            sites_processed_count += 1
            controller_logger.info(f">>> Finished scrape for site: {site_key} successfully. <<<")
        except Exception as e_orchestrate_site:
            controller_logger.error(f"XXX Orchestration for site {site_key} failed: {e_orchestrate_site} XXX", exc_info=True)
            sites_failed_count += 1
            
    return sites_processed_count, sites_failed_count

async def main_controller():
    controller_logger.info("===== Site Controller Initializing =====")
    overall_start_time = datetime.datetime.now()

    # --- NEW: Manage lock file ---
    if SCRAPE_LOCK_FILE.exists():
        controller_logger.warning("Scrape lock file already exists. Another scrape may be running. Aborting.")
        return
    
    try:
        SCRAPE_LOCK_FILE.touch() # Create the lock file
        controller_logger.info("Scrape lock file created.")

        global_scraper_settings, enabled_sites = load_orchestrator_settings()
        if global_scraper_settings is None or enabled_sites is None:
            controller_logger.critical("Exiting: Critical error loading settings.")
            return

        s_succ, s_fail = await run_regular_scrape_orchestration(global_scraper_settings, enabled_sites)
        
        controller_logger.info("\n--------------------------------------------------")
        controller_logger.info(f"REGULAR SCRAPE SUMMARY: Sites Succeeded: {s_succ}, Sites Failed: {s_fail}")
    
    finally:
        # --- NEW: Ensure lock file is removed ---
        if SCRAPE_LOCK_FILE.exists():
            SCRAPE_LOCK_FILE.unlink()
            controller_logger.info("Scrape lock file removed.")
        
        controller_logger.info(f"Total Site Controller Duration: {datetime.datetime.now() - overall_start_time}")
        controller_logger.info("===== Site Controller Finished =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TenFin Site Controller: Handles regular tender scraping.")
    parser.add_argument("--type",type=str,choices=["regular"],default="regular",help="Type of operation.")
    args = parser.parse_args()

    try:
        from database import init_db
        if not init_db():
            controller_logger.critical("Database initialization failed. Aborting scrape.")
            if SCRAPE_LOCK_FILE.exists(): SCRAPE_LOCK_FILE.unlink() # Cleanup on early exit
            sys.exit(1)
    except ImportError:
        print("FATAL (SiteController): Could not import 'init_db' from database.py.")
        sys.exit(1)

    if args.type == "regular":
        asyncio.run(main_controller())
    else:
        print(f"Scrape type '{args.type}' is not supported. Use 'regular'.")
