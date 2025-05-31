# --- START OF dashboard.py (Ensuring completeness and AI Proxy corrections) ---
import os
import re
import io
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set 
import datetime
import logging
import subprocess 
import sys
import time
from urllib.parse import urlparse, urljoin 

from fastapi import FastAPI, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse, FileResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from openpyxl import Workbook
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.datastructures import URL
from pydantic import BaseModel 

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout # Playwright needed for _fetch_rot_captcha_image
import base64 # For encoding CAPTCHA image
import httpx # For AI proxy

# --- Filter Engine Imports (Robust) ---
try:
    # Ensure INDIAN_STATES is also imported if used by filter_engine or dashboard later
    from filter_engine import run_filter, parse_tender_blocks_from_tagged_file, extract_tender_info_from_tagged_block, INDIAN_STATES 
    filter_engine_available = True
except ImportError: 
    print("ERROR: Could not import 'filter_engine' components. Filter and AI data prep functionality may be affected.")
    INDIAN_STATES = ["Error: State List Unavailable"] # Fallback for INDIAN_STATES
    filter_engine_available = False
    # Define dummy functions for filter_engine components if import fails
    def _log_filter_engine_error(message): # Helper to avoid NameError on logger if it's not defined yet
        logger_ref = globals().get('logger')
        if logger_ref and isinstance(logger_ref, logging.Logger):
            logger_ref.error(message)
        else:
            print(f"FILTER_ENGINE_ERROR: {message}")

    def run_filter(*args, **kwargs): 
        _log_filter_engine_error("CRITICAL ERROR: filter_engine.run_filter not imported."); 
        raise RuntimeError("filter_engine.run_filter not available.")
    def parse_tender_blocks_from_tagged_file(*args, **kwargs):
        _log_filter_engine_error("CRITICAL ERROR: filter_engine.parse_tender_blocks_from_tagged_file not imported.");
        raise RuntimeError("filter_engine.parse_tender_blocks_from_tagged_file not available.")
    def extract_tender_info_from_tagged_block(*args, **kwargs):
        _log_filter_engine_error("CRITICAL ERROR: filter_engine.extract_tender_info_from_tagged_block not imported.");
        raise RuntimeError("filter_engine.extract_tender_info_from_tagged_block not available.")

# --- MAIN_SITE_CONFIGS Import (from your original dashboard.py v13 logic) ---
# This is for the "Regular Scrape" part of your dashboard (e.g., populating site lists for settings)
available_sites_main: List[str] = ["Error: MAIN_SITE_CONFIGS not loaded"] 
MAIN_SITE_CONFIGS: Dict[str, Any] = {} 
try:
    from scrape import SITE_CONFIGS as MAIN_SITE_CONFIGS_IMPORT # scrape.py should define this
    if isinstance(MAIN_SITE_CONFIGS_IMPORT, dict) and MAIN_SITE_CONFIGS_IMPORT:
        MAIN_SITE_CONFIGS = MAIN_SITE_CONFIGS_IMPORT
        available_sites_main = sorted(list(MAIN_SITE_CONFIGS.keys()))
        if not available_sites_main: # Should not happen if MAIN_SITE_CONFIGS_IMPORT was valid dict
             print("ERROR: SITE_CONFIGS from scrape.py has no keys after import.")
             available_sites_main = ["Error: Main Config Empty/Invalid"]
    else: 
        print("ERROR: SITE_CONFIGS imported from scrape.py is empty or not a dictionary.")
        available_sites_main = ["Error: Main Config Empty/Invalid"]
except ImportError: 
    print("ERROR: Could not import 'scrape' module for MAIN_SITE_CONFIGS. Regular scrape site list may be affected.")
except Exception as e_main_configs: 
    print(f"ERROR: Could not process SITE_CONFIGS from scrape.py: {e_main_configs}")
    available_sites_main = ["Error: Main Config Import Failed"]


# --- URL_SUFFIX_ROT Import (Corrected for new workflow) ---
try:
    from headless_rot_worker import URL_SUFFIX_ROT # Primary: Your new headless worker
    print("INFO: Using URL_SUFFIX_ROT from 'headless_rot_worker.py'.")
except ImportError:
    print("WARNING: Could not import URL_SUFFIX_ROT from 'headless_rot_worker.py'.")
    try:
        from scrape_rot_advanced import URL_SUFFIX_ROT # Secondary fallback: if you named it this
        print("INFO: Using URL_SUFFIX_ROT from 'scrape_rot_advanced.py'.")
    except ImportError:
        print("WARNING: Could not import URL_SUFFIX_ROT from 'scrape_rot_advanced.py'.")
        try:
            from scrape_rot import URL_SUFFIX_ROT # Tertiary fallback: original name
            print("INFO: Using URL_SUFFIX_ROT from 'scrape_rot.py'.")
        except ImportError:
            print("CRITICAL WARNING: Could not import URL_SUFFIX_ROT from any known ROT script. Using dashboard's hardcoded default. This might lead to issues if the suffix changes in the scraper scripts."); 
            URL_SUFFIX_ROT = "?page=WebTenderStatusLists&service=page" # Dashboard's own fallback
# === CONFIGURATION ===
try: SCRIPT_DIR = Path(__file__).parent.resolve()
except NameError: SCRIPT_DIR = Path('.').resolve()

BASE_PATH = SCRIPT_DIR / "scraped_data"; FILTERED_PATH = BASE_PATH / "Filtered Tenders"; SITE_SPECIFIC_MERGED_PATH = BASE_PATH / "SiteSpecificMerged" 
TEMPLATES_DIR = SCRIPT_DIR / "templates"; LOG_DIR = SCRIPT_DIR / "logs"; SETTINGS_FILE = SCRIPT_DIR / "settings.json"
BASE_PATH_ROT = SCRIPT_DIR / "scraped_data_rot"; FILTERED_PATH_ROT = BASE_PATH_ROT / "Filtered Tenders ROT"; SITE_SPECIFIC_MERGED_PATH_ROT = BASE_PATH_ROT / "SiteSpecificMergedROT"
DOWNLOADS_DIR_ROT_BASE = BASE_PATH_ROT / "DetailDownloads"; LOG_DIR_ROT_ROOT = SCRIPT_DIR / "logs_rot" 
SITE_CONTROLLER_SCRIPT_PATH = SCRIPT_DIR / "site_controller.py"; SCHEDULER_SETUP_SCRIPT_PATH = SCRIPT_DIR / "scheduler_setup.py"
FILTERED_TENDERS_FILENAME = "Filtered_Tenders.json" 

AI_ANALYSIS_DATA_FILE = BASE_PATH_ROT / "ai_analysis_data.json" 
# !!! For Production: Move these to environment variables or a secure, uncommitted config file !!!
OPENWEBUI_API_BASE_URL_CONFIG = os.environ.get("OPENWEBUI_API_BASE_URL", "http://192.168.1.104:8080") 
OPENWEBUI_API_KEY_CONFIG = os.environ.get("OPENWEBUI_API_KEY", "sk-69930a705a334f66b3169980850e197e") # YOUR API KEY
DEFAULT_OLLAMA_MODEL_ID_CONFIG = os.environ.get("DEFAULT_OLLAMA_MODEL_ID", "gemma3:12b") 

TEMP_RUN_DATA_DIR_DASHBOARD = SCRIPT_DIR / "temp_run_data" 
TEMP_RUN_DATA_DIR_DASHBOARD.mkdir(parents=True, exist_ok=True)

DEBUG_SCREENSHOT_DIR = SCRIPT_DIR / "debug_screenshots" 
DEBUG_SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


LOG_DIR.mkdir(parents=True, exist_ok=True); LOG_DIR_ROT_ROOT.mkdir(parents=True, exist_ok=True) 
BASE_PATH.mkdir(parents=True, exist_ok=True); FILTERED_PATH.mkdir(parents=True, exist_ok=True); SITE_SPECIFIC_MERGED_PATH.mkdir(parents=True, exist_ok=True)
BASE_PATH_ROT.mkdir(parents=True, exist_ok=True); FILTERED_PATH_ROT.mkdir(parents=True, exist_ok=True); SITE_SPECIFIC_MERGED_PATH_ROT.mkdir(parents=True, exist_ok=True); DOWNLOADS_DIR_ROT_BASE.mkdir(parents=True, exist_ok=True)
AI_ANALYSIS_DATA_FILE.parent.mkdir(parents=True, exist_ok=True) 

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Dashboard) %(message)s", datefmt="%H:%M:%S")
log_handler = logging.StreamHandler(); log_handler.setFormatter(log_formatter)
logger = logging.getLogger(__name__); logger.setLevel(logging.INFO) 
if not logger.hasHandlers(): logger.addHandler(log_handler); logger.propagate = False

DEFAULT_SETTINGS = {
  "global_scraper_settings": {
    "scrape_py_limits": {"max_list_pages": 100, "retries": 3}, "concurrency": {"list_pages": 5, "detail_pages": 2}, "timeouts": {"page_load": 75000, "detail_page": 75000},
    "rot_scrape_limits": {"max_list_pages": 50, "retries": 2}, "rot_concurrency": {"detail_processing": 2}, 
    "rot_timeouts": {"page_load": 75000, "detail_page": 60000, "download": 180000, "pagination_load_wait": 7000, "element_wait": 20000}
  }, "retention": {"enabled": False,"days": 30 },
  "scheduler": { "main_enabled": False, "main_frequency": "daily", "main_time": "02:00", "main_day_weekly": "7", "main_day_monthly": "1", 
                 "rot_default_status_value": "5" },
  "site_configurations": {}
}
PAGE_LOAD_TIMEOUT_DEFAULT = DEFAULT_SETTINGS["global_scraper_settings"]["timeouts"]["page_load"]
ELEMENT_WAIT_TIMEOUT_DEFAULT = DEFAULT_SETTINGS["global_scraper_settings"]["rot_timeouts"].get("element_wait", 20000)

def load_settings() -> Dict:
    if not SETTINGS_FILE.is_file():
        logger.warning(f"Settings file not found at {SETTINGS_FILE}. Creating with defaults.")
        try:
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(DEFAULT_SETTINGS, f, indent=2)
            logger.info(f"Created default settings file: {SETTINGS_FILE}")
            return DEFAULT_SETTINGS.copy()
        except Exception as e: logger.error(f"Could not create default settings: {e}"); return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f: settings = json.load(f)
        needs_resave = False
        for main_key, default_value in DEFAULT_SETTINGS.items():
            if main_key not in settings:
                settings[main_key] = default_value.copy(); needs_resave = True
            elif isinstance(default_value, dict):
                if not isinstance(settings.get(main_key), dict):
                    settings[main_key] = default_value.copy(); needs_resave = True
                else:
                    for sub_key, sub_default_value in default_value.items():
                        if sub_key not in settings[main_key]:
                            settings[main_key][sub_key] = sub_default_value; needs_resave = True
                        elif isinstance(sub_default_value, dict):
                             if not isinstance(settings[main_key].get(sub_key), dict):
                                 settings[main_key][sub_key] = sub_default_value.copy(); needs_resave = True
                             else:
                                 for deep_key, deep_default in sub_default_value.items():
                                     if deep_key not in settings[main_key][sub_key]:
                                         settings[main_key][sub_key][deep_key] = deep_default; needs_resave = True
        if "scheduler" in settings and isinstance(settings["scheduler"], dict):
            if "paused" in settings["scheduler"]: del settings["scheduler"]["paused"]; needs_resave = True
            for old_rot_key in ["rot_enabled", "rot_frequency", "rot_time", "rot_day_weekly", "rot_day_monthly"]:
                if old_rot_key in settings["scheduler"]: del settings["scheduler"][old_rot_key]; needs_resave = True
            if "rot_default_status_value" not in settings["scheduler"]:
                 settings["scheduler"]["rot_default_status_value"] = DEFAULT_SETTINGS["scheduler"]["rot_default_status_value"]
                 needs_resave = True
        return settings
    except json.JSONDecodeError as e: logger.error(f"Error decoding {SETTINGS_FILE}: {e}. Defaults returned."); return DEFAULT_SETTINGS.copy()
    except Exception as e: logger.error(f"Error reading {SETTINGS_FILE}: {e}. Defaults returned.", exc_info=True); return DEFAULT_SETTINGS.copy()

def save_settings(settings_data: Dict) -> bool:
    try:
        if "scheduler" in settings_data and isinstance(settings_data["scheduler"], dict):
            if "paused" in settings_data["scheduler"]:
                del settings_data["scheduler"]["paused"]; logger.info("Removed 'paused' key from scheduler settings before saving.")
            for key in ["rot_enabled", "rot_frequency", "rot_time", "rot_day_weekly", "rot_day_monthly"]:
                if key in settings_data["scheduler"]:
                    del settings_data["scheduler"][key]
            if "rot_default_status_value" not in settings_data["scheduler"]:
                 settings_data["scheduler"]["rot_default_status_value"] = DEFAULT_SETTINGS["scheduler"]["rot_default_status_value"]
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Settings saved successfully to {SETTINGS_FILE}"); return True
    except Exception as e: logger.error(f"Error saving settings to {SETTINGS_FILE}: {e}"); return False

def format_text_neatly(text: Optional[str], min_len_for_titlecase=8) -> str:
    if not text or not isinstance(text, str): return ""
    cleaned_text = re.sub(r'[^\w\s.,()/\-]', ' ', text); cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    if not cleaned_text: return ""
    if len(cleaned_text) >= min_len_for_titlecase and cleaned_text.isupper() and ' ' not in cleaned_text and cleaned_text.isalnum(): return cleaned_text
    if ' ' in cleaned_text and cleaned_text.isupper(): return cleaned_text.title()
    formatted_text = cleaned_text[0].upper() + cleaned_text[1:].lower(); formatted_text = re.sub(r'(?<=[.!?]\s)([a-z])', lambda m: m.group(1).upper(), formatted_text)
    return formatted_text

app = FastAPI(title="TenFin Tender Dashboard")
templates: Optional[Jinja2Templates] = None
if TEMPLATES_DIR.is_dir():
    try:
        jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(['html', 'xml']))
        jinja_env.filters['neat_format'] = format_text_neatly; templates = Jinja2Templates(env=jinja_env); logger.info("Jinja2 configured.")
    except Exception as e: logger.error(f"Jinja2 init failed: {e}", exc_info=True); templates = None
else: templates = None; logger.error(f"Templates dir '{TEMPLATES_DIR}' not found. UI broken.")

def _validate_subdir(subdir: str, base_dir: Path) -> Path: 
    if not subdir or not isinstance(subdir, str): raise HTTPException(400, "Invalid subdir name.")
    if ".." in subdir or subdir.startswith(("/", "\\")) or any(c in subdir for c in ['<', '>', ':', '"', '|', '?', '*']): raise HTTPException(400, "Invalid subdir format.")
    if not base_dir.is_dir(): raise HTTPException(500, "Server base dir error.")
    try:
        subdir_path_part = Path(subdir)
        if subdir_path_part.is_absolute() or subdir_path_part.drive: raise HTTPException(400, "Subdir must be relative.")
        full_path = (base_dir / subdir_path_part).resolve(); resolved_base_dir = base_dir.resolve()
        if not str(full_path).startswith(str(resolved_base_dir)): raise HTTPException(400, "Path traversal detected.")
        if resolved_base_dir != full_path.parent and resolved_base_dir not in full_path.parents: raise HTTPException(400, "Invalid path nesting.")
    except ValueError as ve: raise HTTPException(400, f"Invalid path: {ve}")
    except HTTPException: raise
    except Exception as e: logger.error(f"Path validation error for '{subdir}': {e}", exc_info=True); raise HTTPException(500, "Internal path error.")
    return full_path

def write_dashboard_status(run_dir: Path, status_message: str): # Renamed to avoid conflict if importing from worker
    # Uses the dashboard's logger
    log = logger 
    try:
        status_file = run_dir / "status.txt"
        status_file.write_text(status_message, encoding='utf-8')
        log.info(f"Dashboard: Status updated to '{status_message}' in {status_file}")
    except Exception as e: 
        log.error(f"Dashboard: Error writing status '{status_message}' to {run_dir}: {e}")

def cleanup_old_filtered_results(): 
    logger.info("Running cleanup for old filtered results (REGULAR and ROT)...")
    settings = load_settings()
    if not settings.get("retention", {}).get("enabled", False): logger.info("Retention cleanup disabled."); return
    try:
        days_to_keep = int(settings.get("retention", {}).get("days", 30)); days_to_keep = max(1, days_to_keep)
        cutoff_timestamp = time.time() - (days_to_keep * 24 * 60 * 60)
        for data_type_label, base_to_clean in [("REGULAR", FILTERED_PATH), ("ROT", FILTERED_PATH_ROT)]:
            deleted_count, kept_count, error_count = 0, 0, 0
            logger.info(f"Cleaning {data_type_label} filtered tenders older than {days_to_keep} days in: {base_to_clean}")
            if not base_to_clean.is_dir(): logger.warning(f"Dir '{base_to_clean}' for {data_type_label} not found."); continue
            for item in base_to_clean.iterdir():
                if item.is_dir():
                    try:
                        item_mtime = item.stat().st_mtime
                        if item_mtime < cutoff_timestamp:
                            logger.info(f"Deleting old {data_type_label} set: {item.name} (Modified: {datetime.datetime.fromtimestamp(item_mtime)})")
                            shutil.rmtree(item); deleted_count += 1
                        else: kept_count += 1
                    except Exception as e: logger.error(f"Error processing dir '{item.name}' ({data_type_label}) for cleanup: {e}"); error_count += 1
            logger.info(f"Retention cleanup for {data_type_label} finished. Deleted: {deleted_count}, Kept: {kept_count}, Errors: {error_count}")
    except ValueError: logger.error("Invalid 'days' in retention settings.")
    except Exception as e: logger.error(f"Error during retention cleanup: {e}", exc_info=True)

# In dashboard.py

def get_rot_site_config(site_key_to_find: str) -> Optional[Dict[str, str]]:
    settings = load_settings()
    all_site_configs = settings.get("site_configurations", {})
    if site_key_to_find in all_site_configs:
        main_config = all_site_configs[site_key_to_find]
        domain_from_settings = main_config.get("domain") 
        
        if domain_from_settings:
            # 1. Clean the domain from settings: remove any trailing '?' or '/'
            cleaned_domain_for_search_url = domain_from_settings.rstrip('/?')

            # 2. URL_SUFFIX_ROT already starts with '?', e.g., "?page=WebTenderStatusLists&service=page"
            #    Simply append it.
            rot_search_url = f"{cleaned_domain_for_search_url}{URL_SUFFIX_ROT}" # This should now be correct
            
            # --- For base_url (used for resolving relative links from scraped pages later) ---
            parsed_cleaned_domain = urlparse(cleaned_domain_for_search_url) # Use the same cleaned domain
            base_path_for_links = parsed_cleaned_domain.path
            
            if not base_path_for_links: 
                base_path_for_links = "/"
            elif not base_path_for_links.endswith('/'):
                base_path_for_links += '/'
            
            base_url_for_resolving_links = f"{parsed_cleaned_domain.scheme}://{parsed_cleaned_domain.netloc}{base_path_for_links}"

            # Heuristic for 'app/' suffix on base_url_for_resolving_links
            if ('nicgep' in base_url_for_resolving_links.lower() or 'eprocure.gov.in' in base_url_for_resolving_links.lower()):
                if not base_url_for_resolving_links.endswith('app/'):
                    path_segments = [s for s in parsed_cleaned_domain.path.split('/') if s]
                    if not path_segments or path_segments[-1].lower() != 'app':
                         base_url_for_resolving_links = urljoin(base_url_for_resolving_links, 'app/')
            
            logger.debug(f"Derived ROT config for '{site_key_to_find}': Search URL='{rot_search_url}', Base URL for links='{base_url_for_resolving_links}'")
            return {"main_search_url": rot_search_url, "base_url": base_url_for_resolving_links}
        else:
            logger.error(f"Domain not found for site_key: {site_key_to_find} in settings.")
    else:
        logger.error(f"Site key '{site_key_to_find}' not found in site_configurations.")
    return None

async def _fetch_rot_captcha_image(site_key: str, tender_status_value: str) -> Optional[str]:
    logger.info(f"CAPTCHA FETCH: Starting for site '{site_key}', status '{tender_status_value}'.")
    rot_config = get_rot_site_config(site_key)
    if not rot_config or not rot_config.get("main_search_url"):
        logger.error(f"CAPTCHA FETCH: No valid ROT URL for site '{site_key}'. Cannot fetch CAPTCHA.")
        return None
    main_search_url = rot_config["main_search_url"]
    logger.debug(f"CAPTCHA FETCH: Derived main_search_url: {main_search_url}")
    settings = load_settings() 
    gs_rot_timeouts = settings.get("global_scraper_settings", {}).get("rot_timeouts", {})
    page_load_timeout = int(gs_rot_timeouts.get("page_load", PAGE_LOAD_TIMEOUT_DEFAULT))
    element_wait_timeout = int(gs_rot_timeouts.get("element_wait", ELEMENT_WAIT_TIMEOUT_DEFAULT))
    async with async_playwright() as p:
        browser = None
        page: Optional[Page] = None 
        try:
            browser = await p.chromium.launch(headless=True) 
            context = await browser.new_context( ignore_https_errors=True, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36" )
            page = await context.new_page()
            logger.info(f"CAPTCHA FETCH: Navigating to {main_search_url} for site '{site_key}'.")
            try:
                await page.goto(main_search_url, timeout=page_load_timeout, wait_until="domcontentloaded")
                logger.info(f"CAPTCHA FETCH: Successfully navigated to URL: {page.url} (Site: {site_key})")
            except PlaywrightTimeout as e_goto:
                logger.error(f"CAPTCHA FETCH: Timeout ({page_load_timeout}ms) during page.goto for '{main_search_url}' (Site: '{site_key}'): {e_goto}")
                if page: await page.screenshot(path=DEBUG_SCREENSHOT_DIR / f"captcha_goto_timeout_{site_key}.png")
                return None
            except Exception as e_goto_other:
                logger.error(f"CAPTCHA FETCH: Error during page.goto for '{main_search_url}' (Site: '{site_key}'): {e_goto_other}")
                if page: await page.screenshot(path=DEBUG_SCREENSHOT_DIR / f"captcha_goto_error_{site_key}.png")
                return None
            logger.info(f"CAPTCHA FETCH: Selecting tender status '{tender_status_value}' for site '{site_key}'.")
            try:
                status_dropdown_locator = page.locator("#tenderStatus")
                await status_dropdown_locator.wait_for(state="attached", timeout=element_wait_timeout // 2)
                await status_dropdown_locator.select_option(value=tender_status_value, timeout=element_wait_timeout // 2)
                selected_value_check = await status_dropdown_locator.evaluate("el => el.value")
                logger.info(f"CAPTCHA FETCH: Tender status dropdown selected. Value is now: '{selected_value_check}' for site '{site_key}'.")
            except PlaywrightTimeout:
                logger.error(f"CAPTCHA FETCH: Timeout selecting/verifying tender status '{tender_status_value}' for site '{site_key}'.")
                await page.screenshot(path=DEBUG_SCREENSHOT_DIR / f"captcha_status_select_timeout_{site_key}.png")
                return None
            except Exception as e_select:
                logger.error(f"CAPTCHA FETCH: Error selecting tender status '{tender_status_value}' for site '{site_key}': {e_select}")
                await page.screenshot(path=DEBUG_SCREENSHOT_DIR / f"captcha_status_select_error_{site_key}.png")
                return None
            logger.debug(f"CAPTCHA FETCH: Waiting a bit after status selection for site '{site_key}'.")
            await page.wait_for_timeout(3000) 
            captcha_img_loc = page.locator("#captchaImage")
            logger.info(f"CAPTCHA FETCH: Attempting to find CAPTCHA image '#captchaImage' for site '{site_key}'.")
            try: 
                await captcha_img_loc.wait_for(state="visible", timeout=element_wait_timeout)
                logger.info(f"CAPTCHA FETCH: Image selector '#captchaImage' is visible for site '{site_key}'.")
            except PlaywrightTimeout: 
                logger.error(f"CAPTCHA FETCH: Image selector '#captchaImage' NOT visible after {element_wait_timeout}ms for site '{site_key}'.")
                await page.screenshot(path=DEBUG_SCREENSHOT_DIR / f"captcha_image_not_visible_{site_key}.png")
                page_html_on_fail = await page.content()
                (DEBUG_SCREENSHOT_DIR / f"captcha_page_html_on_fail_{site_key}.html").write_text(page_html_on_fail, encoding='utf-8')
                logger.debug(f"CAPTCHA FETCH: Saved HTML of failing page to debug_screenshots for site '{site_key}'.")
                return None
            logger.debug(f"CAPTCHA FETCH: Attempting to screenshot the CAPTCHA image element for site '{site_key}'.")
            img_bytes = await captcha_img_loc.screenshot(type='png')
            if not img_bytes:
                logger.error(f"CAPTCHA FETCH: Screenshot of CAPTCHA element resulted in empty bytes for site '{site_key}'.")
                await page.screenshot(path=DEBUG_SCREENSHOT_DIR / f"captcha_empty_bytes_{site_key}.png")
                return None
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            logger.info(f"CAPTCHA FETCH: Successfully captured and Base64 encoded CAPTCHA image for site '{site_key}'.")
            return f"data:image/png;base64,{img_b64}"
        except PlaywrightTimeout as e_outer: 
            logger.error(f"CAPTCHA FETCH: Overall Playwright timeout for site '{site_key}': {e_outer}")
            if page and not page.is_closed():
                 await page.screenshot(path=DEBUG_SCREENSHOT_DIR / f"captcha_overall_timeout_{site_key}.png") 
            return None
        except Exception as e: 
            logger.error(f"CAPTCHA FETCH: General error for site '{site_key}': {e}", exc_info=True)
            if page and not page.is_closed():
                await page.screenshot(path=DEBUG_SCREENSHOT_DIR / f"captcha_general_error_{site_key}.png") 
            return None
        finally:
            if browser: 
                await browser.close()
                logger.debug(f"CAPTCHA FETCH: Playwright browser closed for site '{site_key}'.")

async def _globally_merge_rot_site_files() -> Tuple[int, Optional[Path]]:
    log_prefix = "AI DATA PREP (Global ROT Merge)"
    logger.info(f"--- {log_prefix}: Starting Merge of All Site-Specific ROT Files ---")
    site_specific_rot_merged_dir = BASE_PATH_ROT / "SiteSpecificMergedROT"
    final_global_rot_output_dir = BASE_PATH_ROT
    if not site_specific_rot_merged_dir.is_dir():
        logger.warning(f"{log_prefix}: Site-specific ROT merged directory not found: {site_specific_rot_merged_dir}. No global ROT merge will occur.")
        return 0, None
    file_pattern = "Merged_ROT_*.txt" 
    site_merged_files = list(site_specific_rot_merged_dir.glob(file_pattern))
    if not site_merged_files:
        logger.info(f"{log_prefix}: No '{file_pattern}' files found in {site_specific_rot_merged_dir} for global ROT merge.")
        return 0, None
    logger.info(f"{log_prefix}: Found {len(site_merged_files)} site-specific ROT files for consolidation.")
    global_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_filename_prefix = "Final_ROT_Tender_List_"
    global_final_output_path = final_global_rot_output_dir / f"{output_filename_prefix}{global_timestamp}.txt"
    final_global_rot_output_dir.mkdir(parents=True, exist_ok=True)
    total_globally_unique_tender_blocks = 0
    seen_global_tender_hashes: Set[int] = set()
    try:
        with open(global_final_output_path, "w", encoding="utf-8") as global_outfile:
            for site_file_path in site_merged_files:
                logger.info(f"{log_prefix}: Processing: {site_file_path.name}")
                try:
                    site_file_content = site_file_path.read_text(encoding="utf-8", errors="replace").strip()
                    if not site_file_content:
                        logger.debug(f"{log_prefix}: File {site_file_path.name} is empty. Skipping.")
                        continue
                    raw_blocks = site_file_content.split("--- TENDER END ---")
                    tender_blocks_from_site_file = []
                    for rb in raw_blocks:
                        if "--- TENDER START ---" in rb:
                            full_block = (rb.strip() + "\n--- TENDER END ---").strip()
                            if full_block:
                                tender_blocks_from_site_file.append(full_block)
                    for full_block_text in tender_blocks_from_site_file:
                        block_hash_global = hash(full_block_text) 
                        if block_hash_global not in seen_global_tender_hashes:
                            global_outfile.write(full_block_text + "\n\n") 
                            seen_global_tender_hashes.add(block_hash_global)
                            total_globally_unique_tender_blocks += 1
                except Exception as e_proc_site_file:
                    logger.error(f"{log_prefix}: Error processing site-specific ROT file {site_file_path.name}: {e_proc_site_file}")
        logger.info(f"{log_prefix}: ✅ Merged {total_globally_unique_tender_blocks} unique ROT tender blocks to: {global_final_output_path}")
        return total_globally_unique_tender_blocks, global_final_output_path
    except Exception as e_global_merge:
        logger.error(f"{log_prefix}: Fatal error during global ROT merge: {e_global_merge}", exc_info=True)
        return total_globally_unique_tender_blocks, None

# === ROUTE HANDLERS ===
@app.get("/", response_class=HTMLResponse, name="homepage")
async def homepage(request: Request): 
    if not templates: return HTMLResponse("Template engine error", status_code=503)
    subdirs_regular: List[str] = []
    if FILTERED_PATH.is_dir():
        try: subdirs_regular = sorted([d.name for d in FILTERED_PATH.iterdir() if d.is_dir() and not d.name.startswith('.')])
        except OSError as e: logger.error(f"Error listing regular filtered results {FILTERED_PATH}: {e}")
    else: logger.warning(f"Regular filtered data dir '{FILTERED_PATH}' not found.")
    subdirs_rot: List[str] = []
    if FILTERED_PATH_ROT.is_dir():
        try: subdirs_rot = sorted([d.name for d in FILTERED_PATH_ROT.iterdir() if d.is_dir() and not d.name.startswith('.')])
        except OSError as e: logger.error(f"Error listing ROT filtered results {FILTERED_PATH_ROT}: {e}")
    else: logger.warning(f"ROT filtered data dir '{FILTERED_PATH_ROT}' not found.")
    return templates.TemplateResponse("index.html", {"request": request, "subdirs_eproc": subdirs_regular, "subdirs_rot": subdirs_rot})

@app.get("/settings", name="settings_page", response_class=HTMLResponse)
async def settings_page(request: Request): 
    if not templates: return HTMLResponse("Template engine error", status_code=503)
    current_settings = load_settings()
    return templates.TemplateResponse("settings.html", {"request": request, "settings": current_settings })

@app.get("/rot-manual-scrape", name="rot_manual_scrape_page", response_class=HTMLResponse)
async def rot_manual_scrape_page(request: Request):
    if not templates:
        raise HTTPException(status_code=503, detail="Template engine not configured.")
    settings = load_settings()
    site_configs = settings.get('site_configurations', {})
    sites_available_for_rot = site_configs
    return templates.TemplateResponse("rot_manual_scrape.html", {
        "request": request,
        "sites_for_rot": sites_available_for_rot,
        "settings": settings
    })

@app.get("/rot-ai-analysis", name="rot_ai_analysis_page", response_class=HTMLResponse)
async def rot_ai_analysis_page_route(request: Request):
    if not templates:
        raise HTTPException(status_code=503, detail="Template engine not configured.")
    current_settings = load_settings() 
    return templates.TemplateResponse("rot_ai_analysis_dashboard.html", {
        "request": request,
        "settings": current_settings 
    })

@app.get("/view/{data_type}/{subdir}", name="view_tenders_typed", response_class=HTMLResponse)
async def view_tenders_typed(request: Request, data_type: str, subdir: str):
    if not templates: raise HTTPException(503, "Template engine error.")
    tenders: List[Dict[str, Any]] = []
    if data_type == "rot": base_data_dir_for_view = FILTERED_PATH_ROT
    elif data_type == "regular": base_data_dir_for_view = FILTERED_PATH
    else: raise HTTPException(400, "Invalid data type specified.")
    try: subdir_path = _validate_subdir(subdir, base_dir=base_data_dir_for_view); file_path = subdir_path / FILTERED_TENDERS_FILENAME
    except HTTPException as e: raise e
    except Exception as e: logger.error(f"Path error view '{subdir}' type '{data_type}': {e}", exc_info=True); raise HTTPException(500, "Internal path error.")
    if not subdir_path.is_dir(): raise HTTPException(404, f"Filter set '{subdir}' (Type: {data_type}) not found.")
    if not file_path.is_file(): logger.warning(f"Data file '{FILTERED_TENDERS_FILENAME}' missing in '{subdir_path}'") 
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f: tenders_data = json.load(f)
            if not isinstance(tenders_data, list): logger.error(f"Invalid data (not list) in {file_path}"); raise ValueError("Invalid data.")
            tenders = tenders_data
        except (json.JSONDecodeError, ValueError) as e: logger.error(f"Error parsing {file_path}: {e}"); raise HTTPException(500, f"Error reading data for '{subdir}' (Type: {data_type}).")
        except Exception as e: logger.error(f"Error loading data for '{subdir}' (Type: {data_type}): {e}", exc_info=True); raise HTTPException(500, f"Error loading data for '{subdir}'.")
    subdir_display_name = subdir.replace('_Results', '').replace('_Tenders','').replace('_', ' ')
    template_name = "view_rot.html" if data_type == "rot" else "view.html"
    return templates.TemplateResponse(template_name, {"request": request, "subdir": subdir, "subdir_display_name": subdir_display_name, "tenders": tenders, "data_type": data_type})

@app.get("/tender/{data_type}/{subdir}/{tender_id}", name="view_tender_detail_typed", response_class=HTMLResponse)
async def view_tender_detail_typed(request: Request, data_type: str, subdir: str, tender_id: str):
    if not templates: raise HTTPException(503, "Template engine error.")
    found_tender: Optional[Dict[str, Any]] = None
    if data_type == "rot": base_data_dir_for_detail = FILTERED_PATH_ROT
    elif data_type == "regular": base_data_dir_for_detail = FILTERED_PATH
    else: raise HTTPException(400, "Invalid data type.")
    try: subdir_path = _validate_subdir(subdir, base_dir=base_data_dir_for_detail); file_path = subdir_path / FILTERED_TENDERS_FILENAME
    except HTTPException as e: raise e
    except Exception as e: logger.error(f"Path error detail {subdir}/{tender_id} type {data_type}: {e}"); raise HTTPException(500, "Path error")
    if not subdir_path.is_dir(): raise HTTPException(404, f"Filter set '{subdir}' (Type: {data_type}) not found.")
    if not file_path.is_file(): raise HTTPException(404, f"Data file missing for '{subdir}' (Type: {data_type}).")
    try:
        with open(file_path, "r", encoding="utf-8") as f: tenders_list = json.load(f)
        if not isinstance(tenders_list, list): raise ValueError("Invalid data format.")
        for tender_item in tenders_list:
            if isinstance(tender_item, dict):
                if tender_item.get("primary_tender_id") == tender_id: found_tender = tender_item; break
                if data_type == "regular" and tender_item.get("detail_tender_id") == tender_id: found_tender = tender_item; break
        if not found_tender: raise HTTPException(404, f"Tender ID '{tender_id}' not found in '{subdir}' (Type: {data_type}).")
    except (json.JSONDecodeError, ValueError) as e: logger.error(f"Err reading/parsing {file_path}: {e}"); raise HTTPException(500, "Err reading data.")
    except HTTPException: raise
    except Exception as e: logger.error(f"Err loading detail {subdir}/{tender_id} type {data_type}: {e}", exc_info=True); raise HTTPException(500, "Err loading details.")
    template_name = "tender_detail_rot.html" if data_type == "rot" else "tender_detail.html"
    return templates.TemplateResponse(template_name, {"request": request, "tender": found_tender, "subdir": subdir, "data_type": data_type})

@app.get("/download/{data_type}/{subdir}", name="download_tender_excel_typed", response_class=StreamingResponse)
async def download_tender_excel_typed(data_type: str, subdir: str):
    tenders: List[Dict[str, Any]] = []
    if data_type == "rot": base_data_dir_for_download = FILTERED_PATH_ROT
    elif data_type == "regular": base_data_dir_for_download = FILTERED_PATH
    else: raise HTTPException(400, "Invalid data type.")
    try: subdir_path = _validate_subdir(subdir, base_dir=base_data_dir_for_download); file_path = subdir_path / FILTERED_TENDERS_FILENAME
    except HTTPException as e: raise e
    except Exception as e: logger.error(f"Path err download {subdir} type {data_type}: {e}"); raise HTTPException(500,"Path error")
    if not subdir_path.is_dir() or not file_path.is_file(): raise HTTPException(404, "Data not found.")
    try:
        with open(file_path, "r", encoding="utf-8") as f: tenders_data = json.load(f)
        if not isinstance(tenders_data, list): raise ValueError("Invalid data format.")
        tenders = tenders_data
    except (json.JSONDecodeError, ValueError) as e: logger.error(f"Err reading {file_path} for download: {e}"); raise HTTPException(500, "Err reading data.")
    except Exception as e: logger.error(f"Err download prep {subdir} type {data_type}: {e}", exc_info=True); raise HTTPException(500, "Err prep download.")
    wb = Workbook(); ws = wb.active; safe_sheet_title = re.sub(r'[\\/*?:\[\]]+', '_', subdir.replace('_Results','').replace('_Tenders', ''))[:31]; ws.title = safe_sheet_title if safe_sheet_title else "Tenders"
    if data_type == "rot":
        headers = ["rot_s_no", "rot_tender_id", "rot_title_ref", "rot_organisation_chain", "rot_tender_stage", "source_site_key", "rot_status_detail_page_link", "rot_stage_summary_file_status", "rot_stage_summary_file_path", "rot_stage_summary_filename"]
    else: 
        headers = ["primary_tender_id", "tender_reference_number", "primary_title", "work_description", "organisation_chain", "source_site_key", "tender_type", "tender_category", "form_of_contract", "contract_type", "tender_value", "emd_amount", "tender_fee", "processing_fee", "primary_published_date", "primary_closing_date", "primary_opening_date", "bid_validity_days", "period_of_work_days", "location", "pincode", "nda_pre_qualification", "independent_external_monitor_remarks", "payment_instruments", "covers_info", "tender_documents", "tender_inviting_authority_name", "tender_inviting_authority_address", "detail_page_link"]
    ws.append(headers)
    for tender in tenders:
        row_data = [];
        if not isinstance(tender, dict): continue
        for header in headers:
            value = tender.get(header, "N/A")
            if data_type == "regular" and isinstance(value, list):
                try:
                    if header == "payment_instruments" and value: value = "; ".join([f"{i.get('s_no', '?')}:{i.get('instrument_type', 'N/A')}" for i in value if isinstance(i,dict)])
                    elif header == "covers_info" and value: value = "; ".join([f"{i.get('cover_no','?')}:{i.get('cover_type','N/A')}" for i in value if isinstance(i,dict)])
                    elif header == "tender_documents" and value: value = "; ".join([f"{d.get('name',d.get('link','Doc'))}({d.get('size','?')})" for d in value if isinstance(d, dict)])
                    else: value = ", ".join(map(str, value))
                    if not value: value = "N/A"
                except Exception: value = "[Error Formatting List]"
            elif data_type == "regular" and isinstance(value, dict):
                try: value = json.dumps(value, ensure_ascii=False)
                except Exception: value = "[Error Formatting Dict]"
            if value is None: value = "N/A"
            elif not isinstance(value, (str, int, float, datetime.datetime, datetime.date)): value = str(value)
            row_data.append(value)
        if len(row_data)==len(headers): ws.append(row_data)
        else: logger.warning(f"Col count mismatch ID {tender.get('primary_tender_id' if data_type=='regular' else 'rot_tender_id','??')} in {subdir} (Type: {data_type}).")
    excel_buffer = io.BytesIO(); wb.save(excel_buffer); excel_buffer.seek(0); 
    safe_subdir_name = re.sub(r'[^\w\-]+', '_', subdir); filename_prefix = "ROT_" if data_type == "rot" else ""; filename = f"{filename_prefix}{safe_subdir_name}_Tenders_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    return StreamingResponse(excel_buffer, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename=\"{filename}\""})

@app.post("/bulk-download-typed", name="bulk_download_tender_excel_typed", response_class=StreamingResponse) 
async def bulk_download_tender_excel_typed(request: Request, data_type: str = Form(...), selected_subdirs: List[str] = Form(...)):
    if not selected_subdirs: raise HTTPException(400, "No filter sets selected.")
    logger.info(f"Bulk download requested for {data_type.upper()} sets: {selected_subdirs}")
    if data_type == "rot": base_data_dir_for_bulk = FILTERED_PATH_ROT
    elif data_type == "regular": base_data_dir_for_bulk = FILTERED_PATH
    else: raise HTTPException(400, "Invalid data type for bulk download.")
    wb = Workbook(); wb.remove(wb.active); processed_sheets = 0; errors_bulk = []
    if data_type == "rot":
        headers = ["rot_s_no", "rot_tender_id", "rot_title_ref", "rot_organisation_chain", "rot_tender_stage", "source_site_key", "rot_status_detail_page_link", "rot_stage_summary_file_status", "rot_stage_summary_file_path", "rot_stage_summary_filename"]
    else: 
        headers = ["primary_tender_id", "tender_reference_number", "primary_title", "work_description", "organisation_chain", "source_site_key", "tender_type", "tender_category", "form_of_contract", "contract_type", "tender_value", "tender_fee", "processing_fee", "emd_amount", "primary_published_date", "primary_closing_date", "primary_opening_date", "bid_validity_days", "period_of_work_days", "location", "pincode", "nda_pre_qualification", "independent_external_monitor_remarks", "payment_instruments", "covers_info", "tender_documents", "tender_inviting_authority_name", "tender_inviting_authority_address", "detail_page_link"]
    for subdir_name in selected_subdirs:
        tenders_list_sheet = []
        try:
            subdir_path_validated = _validate_subdir(subdir_name, base_dir=base_data_dir_for_bulk); file_path_to_read = subdir_path_validated / FILTERED_TENDERS_FILENAME
            if not subdir_path_validated.is_dir(): errors_bulk.append(f"Missing Dir: '{subdir_name}'"); continue
            if not file_path_to_read.is_file(): errors_bulk.append(f"Data Missing: '{subdir_name}'"); continue
            with open(file_path_to_read, "r", encoding="utf-8") as f_sheet: tenders_data_sheet = json.load(f_sheet)
            if not isinstance(tenders_data_sheet, list): errors_bulk.append(f"Invalid Data: '{subdir_name}'"); continue
            tenders_list_sheet = tenders_data_sheet; safe_sheet_title_bulk = re.sub(r'[\\/*?:\[\]]+', '_', subdir_name.replace('_Results','').replace('_Tenders',''))[:31]
            ws_bulk = wb.create_sheet(title=safe_sheet_title_bulk if safe_sheet_title_bulk else f"Sheet_{processed_sheets+1}"); ws_bulk.append(headers); rows_added_sheet = 0
            for tender_item_sheet in tenders_list_sheet:
                  row_data_sheet = [];
                  if not isinstance(tender_item_sheet, dict): continue
                  for header_name in headers:
                     value_cell = tender_item_sheet.get(header_name, "N/A")
                     if data_type == "regular" and isinstance(value_cell, list):
                        try:
                            if header_name == "payment_instruments" and value_cell: value_cell = "; ".join([f"{i.get('s_no', '?')}:{i.get('instrument_type', 'N/A')}" for i in value_cell if isinstance(i,dict)])
                            elif header_name == "covers_info" and value_cell: value_cell = "; ".join([f"{i.get('cover_no','?')}:{i.get('cover_type','N/A')}" for i in value_cell if isinstance(i,dict)])
                            elif header_name == "tender_documents" and value_cell: value_cell = "; ".join([f"{d.get('name', d.get('link', 'Doc'))} ({d.get('size','?')})" for d in value_cell if isinstance(d, dict)])
                            else: value_cell = ", ".join(map(str, value_cell))
                            if not value_cell: value_cell = "N/A"
                        except Exception: value_cell = "[Error Formatting List]"
                     elif data_type == "regular" and isinstance(value_cell, dict):
                         try: value_cell = json.dumps(value_cell, ensure_ascii=False)
                         except Exception: value_cell = "[Error Formatting Dict]"
                     if value_cell is None: value_cell = "N/A"
                     elif not isinstance(value_cell, (str, int, float, datetime.datetime, datetime.date)): value_cell = str(value_cell)
                     row_data_sheet.append(value_cell)
                  if len(row_data_sheet)==len(headers): ws_bulk.append(row_data_sheet); rows_added_sheet+=1
                  else: logger.warning(f"Col count mismatch bulk ID {tender_item_sheet.get('primary_tender_id' if data_type=='regular' else 'rot_tender_id','??')} sheet {safe_sheet_title_bulk}.")
            logger.info(f"Added sheet '{safe_sheet_title_bulk}' ({rows_added_sheet} {data_type} items) to bulk download."); processed_sheets += 1
        except HTTPException as http_e_bulk: logger.error(f"Bulk download validation err '{subdir_name}': {http_e_bulk.detail}"); errors_bulk.append(f"Invalid: {subdir_name}")
        except Exception as e_bulk: logger.error(f"Bulk download err for '{subdir_name}': {e_bulk}", exc_info=True); errors_bulk.append(f"Error: {subdir_name}")
    if processed_sheets == 0:
        err_details_bulk = f"Bulk download failed for {data_type.upper()}. No valid sets." + (f" Issues: {'; '.join(errors_bulk)}" if errors_bulk else "")
        logger.error(err_details_bulk); raise HTTPException(400, err_details_bulk)
    if errors_bulk: logger.warning(f"Bulk download ({data_type.upper()}) issues: {'; '.join(errors_bulk)}")
    excel_buffer_bulk = io.BytesIO(); wb.save(excel_buffer_bulk); excel_buffer_bulk.seek(0); 
    timestamp_bulk = datetime.datetime.now().strftime("%Y%m%d_%H%M"); filename_bulk = f"Bulk_{data_type.upper()}_Download_{timestamp_bulk}.xlsx"; 
    logger.info(f"Sending bulk {data_type.upper()} download: {filename_bulk} ({processed_sheets} sheets)")
    return StreamingResponse(excel_buffer_bulk, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename=\"{filename_bulk}\""})

@app.post("/bulk-delete-typed", name="bulk_delete_tender_sets_typed") 
async def bulk_delete_tender_sets_typed(request: Request, data_type: str = Form(...), selected_subdirs: List[str] = Form(...)):
    if not selected_subdirs: raise HTTPException(status_code=400, detail="No filter sets selected.")
    logger.info(f"Bulk delete request for {data_type.upper()} sets: {selected_subdirs}")
    if data_type == "rot": base_data_dir_for_delete = FILTERED_PATH_ROT
    elif data_type == "regular": base_data_dir_for_delete = FILTERED_PATH
    else: raise HTTPException(400, "Invalid data type for bulk delete.")
    deleted_count_bulk = 0; errors_delete = []
    for subdir_name_del in selected_subdirs:
        try:
            subdir_path_to_del = _validate_subdir(subdir_name_del, base_dir=base_data_dir_for_delete)
            if subdir_path_to_del.is_dir(): shutil.rmtree(subdir_path_to_del); logger.info(f"Deleted {data_type.upper()} set: {subdir_path_to_del}"); deleted_count_bulk += 1
            else: logger.warning(f"Dir not found for {data_type.upper()} delete: {subdir_path_to_del} (selected: {subdir_name_del})"); errors_delete.append(f"Not found: {subdir_name_del}")
        except HTTPException as http_e_del: logger.error(f"Validation err bulk delete {data_type.upper()} '{subdir_name_del}': {http_e_del.detail}"); errors_delete.append(f"Invalid: {subdir_name_del}")
        except Exception as e_del: logger.error(f"Error deleting {data_type.upper()} set '{subdir_name_del}': {e_del}", exc_info=True); errors_delete.append(f"Error: {subdir_name_del}")
    redirect_url_path_del = app.url_path_for("homepage"); query_params_del = {}; msg_type_key = f"bulk_delete_status_{data_type}"
    if errors_delete: query_params_del[msg_type_key] = "partial"; query_params_del["errors"] = str(len(errors_delete)); query_params_del["deleted"] = str(deleted_count_bulk)
    elif deleted_count_bulk > 0: query_params_del[msg_type_key] = "ok"; query_params_del["deleted"] = str(deleted_count_bulk)
    else: query_params_del[msg_type_key] = "none_deleted"
    redirect_url_del = URL(redirect_url_path_del).replace_query_params(**query_params_del)
    return RedirectResponse(url=str(redirect_url_del), status_code=status.HTTP_303_SEE_OTHER)

@app.post("/delete/{data_type}/{subdir}", name="delete_tender_set_typed") 
async def delete_tender_set_typed(request: Request, data_type: str, subdir: str):
    if data_type == "rot": base_data_dir_single_del = FILTERED_PATH_ROT
    elif data_type == "regular": base_data_dir_single_del = FILTERED_PATH
    else: raise HTTPException(400, "Invalid data type for delete.")
    try:
        subdir_path_single_del = _validate_subdir(subdir, base_dir=base_data_dir_single_del)
        if subdir_path_single_del.is_dir(): shutil.rmtree(subdir_path_single_del); logger.info(f"Deleted {data_type.upper()} filter set: {subdir_path_single_del}"); return RedirectResponse(url=app.url_path_for('homepage') + f"?msg=filter_set_{data_type}_{subdir}_deleted_successfully", status_code=303)
        else: raise HTTPException(404, f"{data_type.upper()} filter set '{subdir}' not found.")
    except HTTPException as e_single_del_http: raise e_single_del_http
    except Exception as e_single_del: logger.error(f"Error deleting {data_type.upper()} set '{subdir}': {e_single_del}", exc_info=True); raise HTTPException(500, f"Could not delete {data_type.upper()} filter set '{subdir}'.")

@app.get("/run-filter", name="run_filter_form", response_class=HTMLResponse)
async def run_filter_form(request: Request): 
    if not templates: raise HTTPException(503, "Template engine error.")
    if not filter_engine_available: return templates.TemplateResponse("error.html", {"request": request, "error": "Filter engine unavailable."}, status_code=503)
    settings = load_settings(); site_configs = settings.get('site_configurations', {}); available_sites_for_filter = sorted(site_configs.keys()) if site_configs else []
    no_source_file_regular = True; latest_source_filename_regular = None
    if BASE_PATH.is_dir():
        try:
            source_files_reg = sorted([p for p in BASE_PATH.glob("Final_Tender_List_*.txt") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
            if source_files_reg: latest_source_filename_regular = source_files_reg[0].name; no_source_file_regular = False
        except OSError as e: logger.error(f"Error listing regular source files: {e}")
    else: logger.warning(f"Regular source data dir '{BASE_PATH}' missing.")
    no_source_file_rot = True; latest_source_filename_rot = None
    if BASE_PATH_ROT.is_dir():
        try:
            source_files_r = sorted([p for p in BASE_PATH_ROT.glob("Final_ROT_Tender_List_*.txt") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
            if source_files_r: latest_source_filename_rot = source_files_r[0].name; no_source_file_rot = False
        except OSError as e: logger.error(f"Error listing ROT source files: {e}")
    else: logger.warning(f"ROT source data dir '{BASE_PATH_ROT}' missing.")
    return templates.TemplateResponse("run_filter.html", {"request": request, "available_sites": available_sites_for_filter, "no_source_file_regular": no_source_file_regular, "latest_source_filename_regular": latest_source_filename_regular, "no_source_file_rot": no_source_file_rot, "latest_source_filename_rot": latest_source_filename_rot})

@app.post("/run-filter", name="run_filter_submit", response_class=HTMLResponse) 
async def run_filter_submit(request: Request, data_type: str = Form(...), keywords: str = Form(""), regex: bool = Form(False), filter_name: str = Form(...), site_key: str = Form(""), start_date: str = Form(""), end_date: str = Form("")):
    if not templates: raise HTTPException(503, "Template engine error.")
    if not filter_engine_available: return templates.TemplateResponse("error.html", {"request": request, "error": "Filter engine unavailable."}, status_code=503)
    if not filter_name or re.search(r'[<>:"/\\|?*]', filter_name) or ".." in filter_name or filter_name.strip()!=filter_name: return templates.TemplateResponse("error.html", {"request": request, "error": "Invalid Filter Name."}, status_code=400)
    if data_type not in ["regular", "rot"]: return templates.TemplateResponse("error.html", {"request": request, "error": "Invalid data type for filter."}, status_code=400)
    latest_source_file_to_use: Optional[str] = None; base_folder_for_filter: Path
    if data_type == "rot": 
        base_folder_for_filter = BASE_PATH_ROT
        source_file_pattern = "Final_ROT_Tender_List_*.txt" 
    else: 
        base_folder_for_filter = BASE_PATH
        source_file_pattern = "Final_Tender_List_*.txt"
    if base_folder_for_filter.is_dir():
        try:
            source_files_list = sorted([p for p in base_folder_for_filter.glob(source_file_pattern) if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
            if source_files_list: latest_source_file_to_use = source_files_list[0].name
            else: raise FileNotFoundError(f"No '{source_file_pattern}' source file for {data_type} data.")
        except Exception as e_find_source: logger.error(f"Error finding {data_type} source file: {e_find_source}"); return templates.TemplateResponse("error.html", {"request": request, "error": f"Error accessing {data_type} source data."}, status_code=500)
    else: return templates.TemplateResponse("error.html", {"request": request, "error": f"{data_type.capitalize()} source data dir missing."}, status_code=500)
    if not latest_source_file_to_use: return templates.TemplateResponse("error.html", {"request": request, "error": f"No source file for {data_type} filter."}, status_code=500)
    try:
        keyword_list_cleaned = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        result_path_str_actual = run_filter(base_folder=base_folder_for_filter, keywords=keyword_list_cleaned, use_regex=regex, filter_name=filter_name, site_key=site_key or None, start_date=start_date or None, end_date=end_date or None, data_type=data_type)
        if not result_path_str_actual or not Path(result_path_str_actual).is_file(): raise RuntimeError(f"Filter engine failed for {data_type} data.")
        logger.info(f"Filter '{filter_name}' ({data_type.upper()}) OK. Output: {result_path_str_actual}"); 
        created_subdir_name = Path(result_path_str_actual).parent.name 
        return templates.TemplateResponse("success.html", {"request": request, "subdir": created_subdir_name, "data_type": data_type})
    except Exception as e_run_filt: logger.error(f"Filter run error '{filter_name}' ({data_type.upper()}) source '{latest_source_file_to_use}': {e_run_filt}", exc_info=True); return templates.TemplateResponse("error.html", {"request": request, "error": f"Filter failed: {type(e_run_filt).__name__}. Check logs."}, status_code=500)

@app.get("/regex-help", name="regex_help_page", response_class=HTMLResponse) 
async def regex_help_page(request: Request):
    if not templates: raise HTTPException(503, "Template engine error.")
    return templates.TemplateResponse("regex_help.html", {"request": request})

@app.post("/save-retention-settings", name="save_retention_settings") 
async def save_retention_settings(request: Request, enable_retention: bool = Form(False), retention_days: int = Form(30)):
    if retention_days < 1: logger.warning(f"Invalid retention: {retention_days}. Use 1."); retention_days = 1
    elif retention_days > 3650: logger.warning(f"Excessive retention: {retention_days}. Use 3650."); retention_days = 3650
    current_settings = load_settings()
    if "retention" not in current_settings or not isinstance(current_settings.get("retention"), dict): current_settings["retention"] = {}
    current_settings["retention"]["enabled"] = enable_retention; current_settings["retention"]["days"] = retention_days
    if save_settings(current_settings): return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=retention_settings_saved", status_code=303)
    else: return RedirectResponse(url=app.url_path_for('settings_page') + "?error=retention_settings_save_failed", status_code=303)

@app.post("/save-scraper-parameters", name="save_scraper_parameters") 
async def save_scraper_parameters(
    request: Request, concurrency_list: int = Form(...), concurrency_detail: int = Form(...), 
    max_list_pages: int = Form(...), retries: int = Form(...), 
    page_load_timeout: int = Form(...), detail_page_timeout: int = Form(...), 
    rot_max_list_pages: int = Form(DEFAULT_SETTINGS["global_scraper_settings"]["rot_scrape_limits"]["max_list_pages"]), 
    rot_retries: int = Form(DEFAULT_SETTINGS["global_scraper_settings"]["rot_scrape_limits"]["retries"]), 
    rot_detail_processing: int = Form(DEFAULT_SETTINGS["global_scraper_settings"]["rot_concurrency"]["detail_processing"]), 
    rot_page_load_timeout: int = Form(DEFAULT_SETTINGS["global_scraper_settings"]["rot_timeouts"]["page_load"]), 
    rot_detail_page_timeout: int = Form(DEFAULT_SETTINGS["global_scraper_settings"]["rot_timeouts"]["detail_page"]), 
    rot_download_timeout: int = Form(DEFAULT_SETTINGS["global_scraper_settings"]["rot_timeouts"]["download"]), 
    rot_pagination_load_wait: int = Form(DEFAULT_SETTINGS["global_scraper_settings"]["rot_timeouts"]["pagination_load_wait"]),
    rot_default_status_value: str = Form(DEFAULT_SETTINGS["scheduler"]["rot_default_status_value"])
):
    cl = max(1, min(concurrency_list, 50)); cd = max(1, min(concurrency_detail, 20)); mlp = max(1, max_list_pages); rt = max(0, min(retries, 5)); plt = max(10000, page_load_timeout); dpt = max(10000, detail_page_timeout)
    if cl!=concurrency_list or cd!=concurrency_detail or mlp!=max_list_pages or rt!=retries or plt!=page_load_timeout or dpt!=detail_page_timeout: logger.warning("Regular scraper params adjusted.")
    rot_mlp = max(1, rot_max_list_pages); rot_rt = max(0, min(rot_retries, 5)); rot_dp = max(1, min(rot_detail_processing, 10)); rot_plt = max(10000, rot_page_load_timeout); rot_dpt = max(10000, rot_detail_page_timeout); rot_dlt = max(30000, rot_download_timeout); rot_plw = max(1000, rot_pagination_load_wait)
    current_settings = load_settings()
    if "global_scraper_settings" not in current_settings or not isinstance(current_settings["global_scraper_settings"], dict): current_settings["global_scraper_settings"] = {}
    gss = current_settings["global_scraper_settings"]
    if "concurrency" not in gss or not isinstance(gss["concurrency"], dict): gss["concurrency"] = {}
    if "scrape_py_limits" not in gss or not isinstance(gss["scrape_py_limits"], dict): gss["scrape_py_limits"] = {}
    if "timeouts" not in gss or not isinstance(gss["timeouts"], dict): gss["timeouts"] = {}
    gss["concurrency"]["list_pages"] = cl; gss["concurrency"]["detail_pages"] = cd; gss["scrape_py_limits"]["max_list_pages"] = mlp; gss["scrape_py_limits"]["retries"] = rt; gss["timeouts"]["page_load"] = plt; gss["timeouts"]["detail_page"] = dpt
    if "rot_scrape_limits" not in gss or not isinstance(gss["rot_scrape_limits"], dict): gss["rot_scrape_limits"] = {}
    if "rot_concurrency" not in gss or not isinstance(gss["rot_concurrency"], dict): gss["rot_concurrency"] = {}
    if "rot_timeouts" not in gss or not isinstance(gss["rot_timeouts"], dict): gss["rot_timeouts"] = {}
    gss["rot_scrape_limits"]["max_list_pages"] = rot_mlp; gss["rot_scrape_limits"]["retries"] = rot_rt; gss["rot_concurrency"]["detail_processing"] = rot_dp; gss["rot_timeouts"]["page_load"] = rot_plt; gss["rot_timeouts"]["detail_page"] = rot_dpt; gss["rot_timeouts"]["download"] = rot_dlt; gss["rot_timeouts"]["pagination_load_wait"] = rot_plw
    if "scheduler" not in current_settings or not isinstance(current_settings["scheduler"], dict): current_settings["scheduler"] = {}
    current_settings["scheduler"]["rot_default_status_value"] = rot_default_status_value
    if save_settings(current_settings): logger.info(f"Global scraper (REG+ROT) params saved."); return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=scraper_params_saved", status_code=303)
    else: return RedirectResponse(url=app.url_path_for('settings_page') + "?error=scraper_params_save_failed", status_code=303)

@app.post("/save-enabled-sites", name="save_enabled_sites") 
async def save_enabled_sites(request: Request):
    form_data = await request.form()
    current_settings = load_settings()
    if "site_configurations" not in current_settings or not isinstance(current_settings["site_configurations"], dict):
        logger.error("Cannot save: 'site_configurations' key is missing or has an invalid type in settings.json.")
        return RedirectResponse(url=app.url_path_for('settings_page') + "?error=site_config_missing_or_invalid", status_code=303)
    
    updated_count = 0
    # Ensure all known site_keys from settings are processed
    for site_key_from_settings in current_settings["site_configurations"]:
        # Check if the checkbox for this site_key was submitted in the form
        # HTML checkboxes only submit a value if they are checked.
        is_enabled_in_form = form_data.get(f"site_{site_key_from_settings}") == "on"
        
        # Update the 'enabled' status in the current_settings dictionary
        if site_key_from_settings in current_settings["site_configurations"]: # Should always be true here
            current_settings["site_configurations"][site_key_from_settings]["enabled"] = is_enabled_in_form
            updated_count += 1
        else:
            # This case should ideally not happen if looping through keys from current_settings
            logger.warning(f"Site key '{site_key_from_settings}' found in settings but not processed for enabling/disabling. This is unexpected.")

    logger.info(f"Enabled sites settings update processed for {updated_count} sites based on current configuration.")
    
    if save_settings(current_settings):
        logger.info("Enabled sites settings were saved successfully.")
        return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=enabled_sites_saved", status_code=303)
    else:
        logger.error("Failed to save the updated enabled sites settings to settings.json.")
        return RedirectResponse(url=app.url_path_for('settings_page') + "?error=enabled_sites_save_failed", status_code=303)

@app.post("/save-main-schedule", name="save_main_schedule") 
async def save_main_schedule(request: Request, freq_main: str = Form(...), time_main: str = Form("02:00"), day_main_weekly: str = Form("7"), day_main_monthly: str = Form("1")):
    form_data = await request.form(); current_settings = load_settings()
    if "scheduler" not in current_settings or not isinstance(current_settings.get("scheduler"), dict): current_settings["scheduler"] = {}
    is_enabled_main = form_data.get("enable_main_schedule") == "True" 
    current_settings["scheduler"]["main_enabled"] = is_enabled_main
    if is_enabled_main:
        valid_freq = ["daily", "weekly", "monthly"];
        if freq_main not in valid_freq: freq_main = "daily"
        try: datetime.datetime.strptime(time_main, "%H:%M")
        except ValueError: time_main = "02:00"
        valid_days_weekly = [str(i) for i in range(1, 8)];
        if day_main_weekly not in valid_days_weekly: day_main_weekly = "7"
        valid_days_monthly = [str(i) for i in range(1, 29)] + ["last"];
        if day_main_monthly not in valid_days_monthly: day_main_monthly = "1"
        current_settings["scheduler"]["main_frequency"] = freq_main
        current_settings["scheduler"]["main_time"] = time_main
        current_settings["scheduler"]["main_day_weekly"] = day_main_weekly
        current_settings["scheduler"]["main_day_monthly"] = day_main_monthly
    current_settings["scheduler"]["rot_enabled"] = False 
    if save_settings(current_settings): 
        logger.info(f"Schedule settings saved (Main En={is_enabled_main}, ROT cron effectively disabled).")
        return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=schedule_settings_saved_apply_needed", status_code=303)
    else: 
        logger.error("Failed to save schedule settings.")
        return RedirectResponse(url=app.url_path_for('settings_page') + "?error=schedule_settings_save_failed", status_code=303)

@app.post("/apply-schedule", name="apply_schedule") 
async def apply_schedule(request: Request):
    logger.info("Apply schedule request. Executing scheduler_setup.py...")
    if not SCHEDULER_SETUP_SCRIPT_PATH.is_file(): logger.error(f"Scheduler script missing: {SCHEDULER_SETUP_SCRIPT_PATH}"); return RedirectResponse(url=app.url_path_for('settings_page') + "?error=scheduler_script_missing", status_code=303)
    try:
        py_exec = sys.executable ; cmd = [py_exec, str(SCHEDULER_SETUP_SCRIPT_PATH)]; logger.info(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', cwd=SCRIPT_DIR)
        log_out = res.stdout.strip(); log_err = res.stderr.strip();
        if log_out: logger.info(f"Scheduler Setup Output:\n{log_out}")
        if log_err: logger.warning(f"Scheduler Setup Error Output:\n{log_err}")
        if res.returncode == 0: logger.info("scheduler_setup.py OK."); return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=schedule_applied_successfully", status_code=303)
        else: logger.error(f"scheduler_setup.py failed: {res.returncode}."); err_part = log_err.splitlines()[-1] if log_err else "UnknownErr"; safe_err = re.sub(r'[^\w\s-]', '', err_part)[:50]; return RedirectResponse(url=app.url_path_for('settings_page') + f"?error=schedule_apply_failed_{safe_err}", status_code=303)
    except FileNotFoundError: logger.error(f"Failed scheduler_setup.py: Python '{py_exec}' or script not found."); return RedirectResponse(url=app.url_path_for('settings_page') + f"?error=schedule_apply_file_not_found", status_code=303)
    except Exception as e: logger.error(f"Failed scheduler_setup.py: {e}", exc_info=True); return RedirectResponse(url=app.url_path_for('settings_page') + f"?error=schedule_apply_exception_{type(e).__name__}", status_code=303)

@app.post("/run-cleanup", name="run_cleanup_task") 
async def run_cleanup_task(request: Request):
    logger.info("Manual cleanup task triggered.")
    try: cleanup_old_filtered_results(); logger.info("Manual cleanup task finished."); return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=cleanup_finished_successfully", status_code=303)
    except Exception as e: logger.error(f"Manual cleanup error: {e}", exc_info=True); return RedirectResponse(url=app.url_path_for('settings_page') + f"?error=cleanup_failed_{type(e).__name__}", status_code=303)

@app.get("/view-logs", name="view_logs_page", response_class=HTMLResponse) 
async def view_logs_page(request: Request, site_log: Optional[str] = None, log_type: str = "regular"): 
    if not templates: raise HTTPException(503, "Template engine error.")
    log_lines_to_show = 200; controller_log_filename = "site_controller.log"; controller_log_path = LOG_DIR / controller_log_filename
    def get_last_log_lines(log_path_func: Path, num_lines_func: int) -> Optional[str]: 
        if not log_path_func.is_file(): logger.warning(f"Log file not found: {log_path_func}"); return f"[Log file not found: {log_path_func.name}]"
        try:
            with open(log_path_func, 'rb') as f_log:
                f_log.seek(0, os.SEEK_END); end_byte_log = f_log.tell(); lines_to_go_log = num_lines_func; block_size_log = 1024; bytes_read_log = 0; blocks_log = []
                while lines_to_go_log > 0 and bytes_read_log < end_byte_log:
                    read_size_log = min(block_size_log, end_byte_log - bytes_read_log); f_log.seek(-(bytes_read_log + read_size_log), os.SEEK_END); block_data_log = f_log.read(read_size_log); blocks_log.append(block_data_log); bytes_read_log += read_size_log; lines_to_go_log -= block_data_log.count(b'\n') + block_data_log.count(b'\r');
                all_bytes_log = b''.join(reversed(blocks_log)); all_text_log = all_bytes_log.decode('utf-8', errors='replace'); lines_log = all_text_log.splitlines(); last_lines_log = lines_log[-num_lines_func:]; return "\n".join(last_lines_log)
        except Exception as e_log_read: logger.error(f"Error reading log {log_path_func}: {e_log_read}", exc_info=True); return f"[Error reading log '{log_path_func.name}': {e_log_read}]"
    controller_log_content_val = get_last_log_lines(controller_log_path, log_lines_to_show)
    regular_site_log_files = sorted([p.name for p in LOG_DIR.glob("scrape_*.log") if p.is_file() and not p.name.startswith("scrape_rot_")])
    rot_site_log_files = sorted([p.name for p in LOG_DIR_ROT_ROOT.glob("scrape_rot_*.log") if p.is_file()]) 
    selected_site_log_filename_val = None; selected_site_log_content_val = None
    current_log_dir_for_selected = LOG_DIR if log_type == "regular" else LOG_DIR_ROT_ROOT
    if site_log:
        if (log_type == "regular" and site_log in regular_site_log_files) or (log_type == "rot" and site_log in rot_site_log_files):
            selected_site_log_filename_val = site_log; selected_site_log_content_val = get_last_log_lines(current_log_dir_for_selected / site_log, log_lines_to_show)
    return templates.TemplateResponse("view_logs.html", {"request": request, "controller_log_filename": controller_log_filename, "controller_log_content": controller_log_content_val, "regular_site_log_files": regular_site_log_files, "rot_site_log_files": rot_site_log_files, "selected_site_log_filename": selected_site_log_filename_val, "selected_site_log_content": selected_site_log_content_val, "current_log_type_filter": log_type})

@app.get("/download-log/{log_type}/{filename}", name="download_log_typed") 
async def download_log_typed(log_type: str, filename: str):
    is_controller_log = filename == "site_controller.log" and log_type == "controller" 
    is_regular_site_log = log_type == "regular" and filename.startswith("scrape_") and filename.endswith(".log") and not filename.startswith("scrape_rot_")
    is_rot_site_log = log_type == "rot" and filename.startswith("scrape_rot_") and filename.endswith(".log")
    if not (is_controller_log or is_regular_site_log or is_rot_site_log): logger.warning(f"Attempt to download non-allowed log: {log_type}/{filename}"); raise HTTPException(403, "Access denied to this log file type.")
    log_path_dl: Path; resolved_base_log_dir_dl: Path
    if is_controller_log: log_path_dl = (LOG_DIR / filename).resolve(); resolved_base_log_dir_dl = LOG_DIR.resolve()
    elif is_regular_site_log: log_path_dl = (LOG_DIR / filename).resolve(); resolved_base_log_dir_dl = LOG_DIR.resolve()
    elif is_rot_site_log: log_path_dl = (LOG_DIR_ROT_ROOT / filename).resolve(); resolved_base_log_dir_dl = LOG_DIR_ROT_ROOT.resolve()
    else: raise HTTPException(500, "Log type determination error.") 
    if not log_path_dl.is_file(): raise HTTPException(404, f"Log file '{filename}' (Type: {log_type}) not found.")
    if resolved_base_log_dir_dl != log_path_dl.parent: logger.error(f"Path traversal for log download: {log_type}/{filename} -> {log_path_dl}"); raise HTTPException(403, "Forbidden.")
    try: logger.info(f"Serving log download: {log_path_dl}"); return FileResponse(log_path_dl, media_type='text/plain', filename=filename)
    except Exception as e_serve_log: logger.error(f"Error serving log {filename}: {e_serve_log}", exc_info=True); raise HTTPException(500, "Error serving log.")

@app.get("/download-rot-summary/{site_key}/{filename}", name="download_rot_summary_file")
async def download_rot_summary_file(site_key: str, filename: str):
    logger.info(f"Request to download ROT summary file: SiteKey='{site_key}', Filename='{filename}'")
    if not site_key or not filename: logger.warning("Download ROT summary: Missing site_key or filename."); raise HTTPException(status_code=400, detail="Site key and filename are required.")
    def _clean_path_component(component: str) -> str: return re.sub(r'[^\w\-._]', '', component)
    cleaned_site_key = _clean_path_component(site_key); cleaned_filename = _clean_path_component(filename)
    if not cleaned_site_key or not cleaned_filename: logger.warning(f"Download ROT summary: Invalid chars in SK/FN. Original: SK='{site_key}', FN='{filename}'"); raise HTTPException(status_code=400, detail="Invalid site key or filename format.")
    file_path = (DOWNLOADS_DIR_ROT_BASE / cleaned_site_key / cleaned_filename).resolve()
    intended_base_download_dir = (DOWNLOADS_DIR_ROT_BASE / cleaned_site_key).resolve()
    if not str(file_path).startswith(str(intended_base_download_dir)): logger.error(f"Download ROT summary: Path traversal. Req: '{filename}' in '{site_key}', Res: '{file_path}', Base: '{intended_base_download_dir}'"); raise HTTPException(status_code=403, detail="Access forbidden.")
    if not file_path.is_file(): logger.error(f"Download ROT summary: File not found: {file_path}"); raise HTTPException(status_code=404, detail=f"File '{cleaned_filename}' not found for site '{cleaned_site_key}'.")
    try:
        logger.info(f"Serving ROT summary file: {file_path}"); media_type = "application/octet-stream"; file_extension = Path(cleaned_filename).suffix.lower()
        if file_extension == ".pdf": media_type = "application/pdf"
        elif file_extension in [".xls", ".xlsx"]: media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif file_extension == ".html" or file_extension == ".htm": media_type = "text/html"
        return FileResponse(path=file_path, filename=cleaned_filename, media_type=media_type)
    except Exception as e: logger.error(f"Error serving ROT summary file {file_path}: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Error serving file.")

@app.get("/rot/request-captcha/{site_key}", name="rot_request_captcha")
async def rot_request_captcha_endpoint(request: Request, site_key: str, tender_status_value: Optional[str] = None):
    settings = load_settings()
    default_rot_status = settings.get("scheduler", {}).get("rot_default_status_value", "5")
    actual_tender_status_value = request.query_params.get("tender_status_value", default_rot_status) 
    if not site_key: raise HTTPException(status_code=400, detail="Site key is required.")
    if not actual_tender_status_value.isdigit(): 
        logger.warning(f"Invalid tender_status_value '{actual_tender_status_value}' for CAPTCHA request for site {site_key}. Using default '{default_rot_status}'.")
        actual_tender_status_value = default_rot_status 
    captcha_data_uri = await _fetch_rot_captcha_image(site_key, actual_tender_status_value)
    if captcha_data_uri: return {"site_key": site_key, "tender_status_value": actual_tender_status_value, "captcha_image_uri": captcha_data_uri}
    else: logger.error(f"Failed to fetch CAPTCHA for {site_key}, status {actual_tender_status_value}."); raise HTTPException(status_code=500, detail=f"Could not fetch CAPTCHA for {site_key}.")

# In dashboard.py

@app.post("/rot/initiate-rot-worker/{site_key}", name="initiate_rot_worker") # Renamed for clarity
async def initiate_rot_worker_endpoint(request: Request, site_key: str, tender_status_value: str = Form(...)):
    logger.info(f"Request to INITIATE ROT WORKER: Site='{site_key}', Status='{tender_status_value}'")
    if not all([site_key, tender_status_value]):
        raise HTTPException(status_code=400, detail="Site key and tender status are required.")
    
    worker_script_name = "headless_rot_worker.py" 
    worker_script_path = SCRIPT_DIR / worker_script_name
    
    if not worker_script_path.is_file():
        logger.error(f"ROT Worker script '{worker_script_name}' not found at {worker_script_path}.")
        raise HTTPException(status_code=500, detail="ROT worker script component is missing on the server.")

    try:
        python_exec = sys.executable
        cleaned_site_key_for_id = re.sub(r'[^\w\-]+', '_', site_key) 
        # Generate a unique run_id for this worker instance
        run_id = f"rot_worker_{cleaned_site_key_for_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Create the temporary run directory for this worker
        run_dir = TEMP_RUN_DATA_DIR_DASHBOARD / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        # Initialize status
        (run_dir / "status.txt").write_text("INITIATED", encoding='utf-8')

        command = [
            python_exec, str(worker_script_path),
            "--site_key", site_key, # Argument name expected by headless_rot_worker.py
            "--tender_status", tender_status_value,
            "--run_id", run_id 
            # No --captcha_solution here; worker will save image and wait for answer file
        ]

        logger.info(f"Launching ROT Worker command: {' '.join(command)}")
        # Run as a background process. Dashboard does not wait for it to complete.
        process = subprocess.Popen(command, cwd=SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        logger.info(f"Launched background ROT Worker for {site_key} (PID: {process.pid}). Run ID: {run_id}")
        
        # The client (rot_manual_scrape.html) will now poll /get-captcha-status/{run_id}
        # and then POST the answer to /submit-captcha-answer/{run_id}
        response_content = {
            "message": f"ROT Worker ({worker_script_name}) initiated for site '{site_key}'. Waiting for CAPTCHA.",
            "run_id": run_id, 
            "site_key": site_key,
            # The UI will use this run_id to poll for the captcha image
        }
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=response_content)
    except Exception as e: 
        logger.error(f"Failed to launch {worker_script_name} for site {site_key}: {e}", exc_info=True); 
        raise HTTPException(status_code=500, detail=f"Failed to start ROT worker for {site_key}: {type(e).__name__}")

@app.post("/run-scraper-now", name="run_scraper_now") 
async def run_scraper_now(request: Request, scrape_type: str = Form("regular")): 
    logger.info(f"Manual 'Run Scraper Now' ({scrape_type}) triggered via dashboard.")
    if scrape_type == "rot": # This button should ideally not offer "rot" directly anymore if using the dedicated page
        logger.info("ROT scraping is manual via the dedicated 'Manual ROT Scrape' page. Ignoring general 'Fetch ROT'.")
        return RedirectResponse(url=app.url_path_for('homepage') + "?msg=manual_scrape_rot_is_manual_via_dedicated_page", status_code=303)
    
    scrape_type_to_run = "regular" # "all" and "regular" both trigger regular scrape
    message_key = "manual_scrape_regular_triggered"
    if scrape_type == "all":
        logger.info("'Fetch All' (deprecated) now only triggers REGULAR scrape. ROT is manual.")
        message_key = "manual_scrape_all_triggers_regular_only"


    if not SITE_CONTROLLER_SCRIPT_PATH.is_file(): 
        logger.error(f"Site controller script not found: {SITE_CONTROLLER_SCRIPT_PATH}")
        return RedirectResponse(url=app.url_path_for('homepage') + "?error=controller_script_missing", status_code=303)
    try:
        python_executable_val = sys.executable
        command_list = [python_executable_val, str(SITE_CONTROLLER_SCRIPT_PATH), "--type", scrape_type_to_run]
        logger.info(f"Executing command in background: {' '.join(command_list)}")
        process_bg = subprocess.Popen(command_list, cwd=SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        logger.info(f"Launched site_controller.py (Type: {scrape_type_to_run}) BG (PID: {process_bg.pid}). Check logs.")
        return RedirectResponse(url=app.url_path_for('homepage') + f"?msg={message_key}", status_code=303)
    except FileNotFoundError: 
        logger.error(f"Failed execute site_controller.py: Python '{python_executable_val}' or script missing.")
        return RedirectResponse(url=app.url_path_for('homepage') + f"?error=manual_scrape_file_not_found_{scrape_type_to_run}", status_code=303)
    except Exception as e_launch: 
        logger.error(f"Failed launch site_controller.py (Type: {scrape_type_to_run}): {e_launch}", exc_info=True)
        return RedirectResponse(url=app.url_path_for('homepage') + f"?error=manual_scrape_launch_failed_{scrape_type_to_run}_{type(e_launch).__name__}", status_code=303)

# === AI Analysis Backend Endpoints ===
@app.post("/rot-prepare-analysis-data", name="rot_prepare_analysis_data")
async def rot_prepare_analysis_data_route():
    logger.info("AI Data Preparation: Received request to collate ROT data.")
    if not filter_engine_available:
        logger.error("AI Data Preparation: Filter engine components are not available.")
        raise HTTPException(status_code=500, detail="Core data processing components (filter_engine) are missing.")
    try:
        merged_count, latest_final_rot_file_path = await _globally_merge_rot_site_files()
        if latest_final_rot_file_path is None or not latest_final_rot_file_path.is_file():
            message_detail = "No site-specific ROT data found to merge globally."
            if merged_count == 0 and latest_final_rot_file_path is None : 
                 logger.warning(f"AI Data Preparation: {message_detail}")
            else: 
                 message_detail = "Global ROT data merge failed or produced no output file."
                 logger.error(f"AI Data Preparation: {message_detail}")
            raise HTTPException(status_code=404, detail=f"{message_detail} Ensure 'Merged_ROT_SITEKEY_DATE.txt' files exist in 'scraped_data_rot/SiteSpecificMergedROT/'.")

        logger.info(f"AI Data Preparation: Using globally merged source file: {latest_final_rot_file_path.name}")
        tagged_blocks = parse_tender_blocks_from_tagged_file(latest_final_rot_file_path)
        collated_rot_tenders: List[Dict[str, Any]] = []
        if tagged_blocks:
            for block_text in tagged_blocks:
                try:
                    tender_info = extract_tender_info_from_tagged_block(block_text)
                    if tender_info.get("data_type") == "rot":
                        collated_rot_tenders.append(tender_info)
                except Exception as e_parse_block:
                    logger.error(f"AI Data Preparation: Error processing a tender block: {e_parse_block}", exc_info=False)
        AI_ANALYSIS_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(AI_ANALYSIS_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(collated_rot_tenders, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"AI Data Preparation: Collated {len(collated_rot_tenders)} ROT items from '{latest_final_rot_file_path.name}' into {AI_ANALYSIS_DATA_FILE}")
        return JSONResponse(
            content={
                "message": f"ROT data collated from '{latest_final_rot_file_path.name}'. {len(collated_rot_tenders)} items processed.",
                "file_path": str(AI_ANALYSIS_DATA_FILE.relative_to(SCRIPT_DIR))
            }, status_code=status.HTTP_200_OK )
    except HTTPException as http_exc:
        logger.error(f"AI Data Preparation HTTP Error: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"AI Data Preparation Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to prepare ROT data: {type(e).__name__}")

@app.get("/get-collated-rot-data-json", name="get_collated_rot_data_json")
async def get_collated_rot_data_json_route():
    logger.info("AI Data Request: Serving collated ROT data.")
    if not AI_ANALYSIS_DATA_FILE.is_file():
        logger.warning(f"AI Data Request: Collated file not found: {AI_ANALYSIS_DATA_FILE}")
        raise HTTPException(status_code=404, detail="Collated ROT data not prepared. Run 'Prepare Data' first.")
    try:
        with open(AI_ANALYSIS_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"AI Data Request: Error serving collated data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to serve collated ROT data.")

class AIPromptRequest(BaseModel):
    model: Optional[str] = None
    prompt: str

@app.post("/proxy-ai-chat", name="proxy_ai_chat")
async def proxy_ai_chat_route(ai_request: AIPromptRequest):
    logger.info(f"AI Proxy: Chat request. Model: {ai_request.model or DEFAULT_OLLAMA_MODEL_ID_CONFIG}")
    target_model = ai_request.model if ai_request.model else DEFAULT_OLLAMA_MODEL_ID_CONFIG
    openwebui_chat_endpoint = f"{OPENWEBUI_API_BASE_URL_CONFIG.rstrip('/')}/api/chat/completions" 
    payload = {"model": target_model, "messages": [{"role": "user", "content": ai_request.prompt}], "stream": False}
    headers = {'Content-Type': 'application/json'}
    if OPENWEBUI_API_KEY_CONFIG: 
        headers['Authorization'] = f'Bearer {OPENWEBUI_API_KEY_CONFIG}'
    logger.debug(f"AI Proxy: Sending to OpenWebUI. Endpoint: {openwebui_chat_endpoint}, Model: {target_model}")
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(openwebui_chat_endpoint, json=payload, headers=headers)
            response.raise_for_status() 
            ai_response_data = response.json()
            logger.info(f"AI Proxy: Received response from OpenWebUI (Status: {response.status_code}).")
            return JSONResponse(content=ai_response_data)
    except httpx.HTTPStatusError as exc:
        error_content_for_logging = "Raw text or unknown structure from AI service error."
        error_detail_for_fastapi = "AI service returned an error."
        try:
            error_content_json = exc.response.json()
            if isinstance(error_content_json, dict) and "error" in error_content_json:
                if isinstance(error_content_json["error"], dict) and "message" in error_content_json["error"]:
                    error_content_for_logging = error_content_json["error"]["message"]
                    error_detail_for_fastapi = error_content_json["error"]["message"]
                elif isinstance(error_content_json["error"], str):
                    error_content_for_logging = error_content_json["error"]
                    error_detail_for_fastapi = error_content_json["error"]
                else:
                    error_content_for_logging = str(error_content_json) 
                    error_detail_for_fastapi = str(error_content_json)
            elif isinstance(error_content_json, dict) and "detail" in error_content_json: 
                error_content_for_logging = error_content_json["detail"]
                error_detail_for_fastapi = error_content_json["detail"]
            else: 
                error_content_for_logging = str(error_content_json)
                error_detail_for_fastapi = "Error detail not in expected format from AI service."
        except json.JSONDecodeError: 
            error_content_for_logging = exc.response.text
            error_detail_for_fastapi = exc.response.text[:200] 
        logger.error(f"AI Proxy: HTTP Error from OpenWebUI: {exc.response.status_code} - Detail: {error_content_for_logging}")
        raise HTTPException(status_code=exc.response.status_code, detail=str(error_detail_for_fastapi))
    except httpx.RequestError as exc:
        logger.error(f"AI Proxy: Request Error to OpenWebUI: {exc}")
        raise HTTPException(status_code=503, detail=f"AI service connection error: {type(exc).__name__}")
    except Exception as e:
        logger.error(f"AI Proxy: Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI proxy unexpected error: {type(e).__name__}")

# === Endpoints for scraper2.py CAPTCHA Interaction ===
@app.get("/solve-captcha/{run_id}", name="solve_captcha_page", response_class=HTMLResponse)
async def solve_captcha_page_route(request: Request, run_id: str):
    if not templates:
        raise HTTPException(status_code=503, detail="Template engine not configured.")
    if not re.match(r"^[a-zA-Z0-9_.-]+$", run_id) or ".." in run_id:
        raise HTTPException(status_code=400, detail="Invalid run_id format.")
    return templates.TemplateResponse("solve_captcha.html", {"request": request, "run_id": run_id})

@app.get("/get-captcha-status/{run_id}", name="get_captcha_status")
async def get_captcha_status_route(run_id: str):
    if not re.match(r"^[a-zA-Z0-9_.-]+$", run_id) or ".." in run_id: # Basic security
        logger.warning(f"Dashboard: Invalid run_id format in get_captcha_status: {run_id}")
        raise HTTPException(status_code=400, detail="Invalid run_id format.")
    
    # Sanitize run_id before using it to construct a path to prevent directory traversal
    safe_run_id_name = Path(run_id).name 
    run_dir = TEMP_RUN_DATA_DIR_DASHBOARD / safe_run_id_name

    status_file = run_dir / "status.txt"
    captcha_b64_file = run_dir / "captcha.b64" # headless_rot_worker.py saves base64 CAPTCHA data here

    response_data = {
        "run_id": run_id, 
        "status": "pending_worker_init", # Default initial status if no status file yet
        "message": "Waiting for worker to initialize and generate CAPTCHA...", 
        "image_data": None
    }

    if not run_dir.is_dir():
        response_data["status"] = "error_run_dir_not_found"
        response_data["message"] = "Error: Worker run directory not found. The run may have ended, errored before creating its directory, or the run ID is incorrect."
        logger.warning(f"Dashboard: Run directory '{run_dir}' not found for run_id '{run_id}' in get_captcha_status.")
        return JSONResponse(content=response_data, status_code=status.HTTP_404_NOT_FOUND)

    if status_file.is_file():
        try:
            current_status_msg = status_file.read_text(encoding='utf-8').strip()
            response_data["message"] = f"Worker status: {current_status_msg}" # Provide current worker status
            
            if current_status_msg == "WAITING_CAPTCHA":
                response_data["status"] = "ready_for_captcha" # UI can use this to show input
                response_data["message"] = "CAPTCHA image ready. Please view and solve."
                if captcha_b64_file.is_file():
                    try:
                        b64_data = captcha_b64_file.read_text(encoding='utf-8').strip()
                        if b64_data: # Ensure data is not empty
                            response_data["image_data"] = f"data:image/png;base64,{b64_data}"
                        else:
                            logger.warning(f"Dashboard: captcha.b64 file is empty for run_id '{run_id}'. Worker might still be writing.")
                            response_data["status"] = "processing_captcha_file"; # Keep polling
                            response_data["message"] = "Worker is generating CAPTCHA image data... please wait."
                    except Exception as e_read_b64:
                        logger.error(f"Dashboard: Error reading captcha.b64 for run_id '{run_id}': {e_read_b64}")
                        response_data["status"] = "error_reading_captcha_file"; 
                        response_data["message"] = "Error: Could not read CAPTCHA image data from worker."
                else:
                    # status.txt says WAITING_CAPTCHA, but .b64 file not yet there
                    logger.debug(f"Dashboard: Status is WAITING_CAPTCHA but captcha.b64 not found yet for run_id '{run_id}'. Worker might still be saving it.")
                    response_data["status"] = "processing_captcha_file"; # Keep polling
                    response_data["message"] = "Worker is preparing CAPTCHA image file... please wait."
            elif current_status_msg == "TIMEOUT_CAPTCHA_INPUT" or "ERROR" in current_status_msg.upper():
                response_data["status"] = "worker_error"; 
                response_data["message"] = f"Worker Error: {current_status_msg}"
            elif "FINISHED_SUCCESS" in current_status_msg.upper() or "FINISHED_NO_DATA" in current_status_msg.upper():
                response_data["status"] = "worker_finished"; 
                response_data["message"] = f"Worker run completed: {current_status_msg}."
            elif current_status_msg == "INITIATED_BY_DASHBOARD" or current_status_msg == "FETCHING_CAPTCHA" or current_status_msg == "WORKER_STARTED":
                 response_data["status"] = "pending_worker_action"; # UI can show a generic "processing"
                 response_data["message"] = f"Worker status: {current_status_msg}... please wait."
            else: # Other intermediate statuses from worker
                response_data["status"] = "worker_processing"; 
                response_data["message"] = f"Worker status: {current_status_msg}"
        except Exception as e_read_status:
            logger.error(f"Dashboard: Error reading status.txt for run_id '{run_id}': {e_read_status}")
            response_data["status"] = "error_reading_status_file"; 
            response_data["message"] = "Error: Could not read worker status file."
    else:
        # Status file not yet created by worker, but run_dir exists. Worker is initializing.
        logger.debug(f"Dashboard: status.txt not yet found for run_id '{run_id}'. Worker process might be starting.")
        # Keep default response_data status as "pending_worker_init"
        response_data["message"] = "Worker process is initializing... please wait." 
    
    return JSONResponse(content=response_data)

@app.post("/submit-captcha-answer/{run_id}", name="submit_captcha_answer")
async def submit_captcha_answer_route(run_id: str, captcha_text: str = Form(...)):
    # Validate run_id format
    if not re.match(r"^[a-zA-Z0-9_.-]+$", run_id) or ".." in run_id:
        logger.warning(f"Dashboard: Invalid run_id format in submit_captcha_answer: {run_id}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid run_id format.")
    
    logger.info(f"Dashboard: Received CAPTCHA answer for run_id '{run_id}': '{captcha_text}'")
    
    safe_run_id_name = Path(run_id).name
    run_dir = TEMP_RUN_DATA_DIR_DASHBOARD / safe_run_id_name
    answer_file = run_dir / "answer.txt"
    status_file = run_dir / "status.txt"

    if not run_dir.is_dir():
        logger.error(f"Dashboard: Cannot submit CAPTCHA answer. Run directory '{run_dir}' not found for run_id '{run_id}'.")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": f"Worker run '{run_id}' not found. It may have already finished or errored early."}
        )
    
    try:
        if status_file.is_file():
            current_worker_status = status_file.read_text(encoding='utf-8').strip().upper()
            terminal_states = ["FINISHED", "ERROR", "TIMEOUT_CAPTCHA_INPUT"]
            if any(term_status in current_worker_status for term_status in terminal_states):
                logger.warning(f"Dashboard: Attempted to submit CAPTCHA for already terminated run '{run_id}'. Current Status: '{current_worker_status}'")
                return JSONResponse(
                    status_code=status.HTTP_409_CONFLICT, 
                    content={"detail": f"Worker run '{run_id}' is already in a terminal state ({current_worker_status}). Cannot submit new CAPTCHA."}
                )
        
        answer_file.write_text(captcha_text, encoding='utf-8')
        
        # Call the new helper function defined within dashboard.py
        write_dashboard_status(run_dir, "CAPTCHA_ANSWER_SUBMITTED_BY_DASHBOARD") 
        
        logger.info(f"Dashboard: Successfully wrote answer to '{answer_file}' for run_id '{run_id}' and updated status.")
        
        return JSONResponse(
            content={"message": f"CAPTCHA answer for run_id '{run_id}' submitted. Worker will attempt to resume scraping.", "run_id": run_id},
            status_code=status.HTTP_200_OK 
        )
    except Exception as e:
        logger.error(f"Dashboard: Error writing answer file or updating status for run_id '{run_id}': {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Failed to submit CAPTCHA answer to worker due to a server error."}
        )


@app.post("/rot/initiate-rot-worker/{site_key}", name="initiate_rot_worker")
async def initiate_rot_worker_endpoint(request: Request, site_key: str, tender_status_value: str = Form(...)):
    logger.info(f"Dashboard: Initiating ROT Worker for Site='{site_key}', Status='{tender_status_value}'")
    if not all([site_key, tender_status_value]):
        logger.error("Dashboard: Missing site_key or tender_status_value for worker initiation.")
        raise HTTPException(status_code=400, detail="Site key and tender status value are required.")
    
    worker_script_name = "headless_rot_worker.py" # This is the script we are building
    worker_script_path = SCRIPT_DIR / worker_script_name
    
    if not worker_script_path.is_file():
        logger.error(f"Dashboard: ROT Worker script '{worker_script_name}' not found at {worker_script_path}.")
        raise HTTPException(status_code=500, detail=f"ROT worker script '{worker_script_name}' component is missing on the server.")

    try:
        python_exec = sys.executable
        # Generate a unique run_id for this worker instance
        cleaned_site_key_for_id = re.sub(r'[^\w\-]+', '_', site_key) 
        run_id = f"rot_worker_{cleaned_site_key_for_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Create the temporary run directory for this worker
        run_dir = TEMP_RUN_DATA_DIR_DASHBOARD / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize status file
        (run_dir / "status.txt").write_text("INITIATED_BY_DASHBOARD", encoding='utf-8')
        logger.info(f"Dashboard: Created run directory '{run_dir}' and status file for Run ID: {run_id}")

        command = [
            python_exec, str(worker_script_path),
            "--site_key", site_key,
            "--tender_status", tender_status_value,
            "--run_id", run_id 
        ]

        logger.info(f"Dashboard: Launching ROT Worker command: {' '.join(command)}")
        # Run as a background process. Dashboard does not wait for it to complete.
        # Ensure appropriate error handling for Popen in a production environment (e.g., permissions)
        process = subprocess.Popen(command, cwd=SCRIPT_DIR, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                   text=True, encoding='utf-8')
        logger.info(f"Dashboard: Launched background ROT Worker for {site_key} (PID: {process.pid}). Run ID: {run_id}")
        
        response_content = {
            "message": f"ROT Worker ({worker_script_name}) initiated for site '{site_key}'. Polling for CAPTCHA image will begin.",
            "run_id": run_id, 
            "site_key": site_key
        }
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=response_content)
    except Exception as e: 
        logger.error(f"Dashboard: Failed to launch {worker_script_name} for site {site_key}: {e}", exc_info=True); 
        raise HTTPException(status_code=500, detail=f"Failed to start ROT worker for {site_key}: {type(e).__name__}")


# --- END OF dashboard.py ---
