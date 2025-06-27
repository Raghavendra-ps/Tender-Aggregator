#!/usr/bin/env python3
# File: eligibility_worker.py

import os
import asyncio
import base64
import argparse
import datetime
from pathlib import Path
import re
import logging
import json
from typing import Optional
import zipfile
import sys
from urllib.parse import urljoin

from playwright.async_api import async_playwright, Page, TimeoutError as PlaywrightTimeout
import httpx

# --- Add project root to path for imports ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

# --- Local Imports ---
from database import SessionLocal, CompanyProfile, EligibilityCheck
from utils.document_processor import extract_text_from_pdf

# --- Directory & Config Constants ---
SITE_DATA_ROOT = PROJECT_ROOT / "site_data"
ELIGIBILITY_DATA_DIR = SITE_DATA_ROOT / "ELIGIBILITY"
TEMP_DATA_DIR = SITE_DATA_ROOT / "TEMP"
ELIGIBILITY_WORKER_RUNS_DIR = TEMP_DATA_DIR / "EligibilityRuns"
LOGS_BASE_DIR_WORKER = PROJECT_ROOT / "LOGS" / "eligibility_worker"
SCRIPT_NAME_TAG = Path(__file__).stem
DASHBOARD_URL = os.environ.get("TENFIN_DASHBOARD_URL", "http://localhost:8081")
TWO_CAPTCHA_API_KEY = os.environ.get("TWO_CAPTCHA_API_KEY", "YOUR_2CAPTCHA_API_KEY_HERE")

# --- Logging ---
logger = logging.getLogger(f"{SCRIPT_NAME_TAG}_placeholder")

def setup_worker_logging(run_id: str):
    global logger
    logger_name = f"{SCRIPT_NAME_TAG}_{run_id[-6:]}"
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers(): logger.handlers.clear(); logger.propagate = False
    
    logger.setLevel(logging.INFO)
    LOGS_BASE_DIR_WORKER.mkdir(parents=True, exist_ok=True)
    log_file_path = LOGS_BASE_DIR_WORKER / f"worker_{run_id}.log"
    
    fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter(f"%(asctime)s [%(levelname)s] (EligWorker-{run_id[-6:]}) %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(f"[EligWorker-{run_id[-6:]}] %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    logger.info(f"Eligibility worker logging initialized. Log file: {log_file_path}")

def update_status(run_dir: Path, status: str, message: Optional[str] = None):
    try:
        status_data = {"status": status, "message": message or status.replace('_', ' ').title()}
        (run_dir / "status.json").write_text(json.dumps(status_data), encoding='utf-8')
        logger.info(f"Status updated to: {status}")
    except Exception as e:
        logger.error(f"Error writing status '{status}': {e}")

async def wait_for_captcha_solution(run_dir: Path) -> Optional[str]:
    answer_file = run_dir / "answer.txt"
    logger.info(f"Waiting for CAPTCHA solution in {answer_file} (max 300s)...")
    for _ in range(300):
        if answer_file.is_file():
            try:
                solution = answer_file.read_text(encoding='utf-8').strip()
                if solution:
                    logger.info("CAPTCHA solution received.")
                    return solution
            except Exception as e:
                logger.error(f"Error reading answer file: {e}")
                return None
        await asyncio.sleep(1)
    logger.warning("Timeout waiting for CAPTCHA solution.")
    return None

def find_primary_document(directory: Path) -> Optional[Path]:
    keywords_by_priority = ["nit", "tenderdocument", "tender_document", "notice"]
    files = list(directory.glob("*.pdf"))
    for keyword in keywords_by_priority:
        for file in files:
            if keyword in file.name.lower():
                logger.info(f"Identified primary document by keyword '{keyword}': {file.name}")
                return file
    if files:
        largest_file = max(files, key=lambda f: f.stat().st_size)
        logger.info(f"No keyword match. Identified largest PDF as primary document: {largest_file.name}")
        return largest_file
    return None

async def handle_popups_and_timeouts(page: Page):
    try:
        important_message_popup_close_button = page.locator("div#msgbox button:has-text('Close')")
        if await important_message_popup_close_button.is_visible(timeout=3000):
            logger.info("ℹ️ 'Important Message' popup detected. Clicking 'Close'.")
            await important_message_popup_close_button.click()
            await page.wait_for_timeout(1000)
    except PlaywrightTimeout:
        logger.debug("No 'Important Message' popup found.")

    try:
        page_html_content = await page.content()
        if "Your session has timed out" in page_html_content:
            logger.info("ℹ️ Session timeout page detected. Clicking restart.")
            await page.locator("a:has-text('restart')").click()
            await page.wait_for_load_state("domcontentloaded", timeout=60000)
    except Exception as e:
        logger.debug(f"Could not check for session timeout message: {e}")
        
async def update_db_status(tender_pk_id: int, status: str, result: Optional[dict] = None, score: Optional[int] = -1):
    db = SessionLocal()
    try:
        check = db.query(EligibilityCheck).filter(EligibilityCheck.tender_id_fk == tender_pk_id).first()
        if not check:
            check = EligibilityCheck(tender_id_fk=tender_pk_id)
            db.add(check)
        check.status = status
        if result:
            check.analysis_result_json = result
        if score is not None:
            check.eligibility_score = score
        db.commit()
        logger.info(f"Updated DB eligibility check for tender {tender_pk_id}: status='{status}', score={score}.")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update DB eligibility check status for tender {tender_pk_id}: {e}")
    finally:
        db.close()

# --- Main Worker Logic ---
async def run_eligibility_check(run_id: str, tender_pk_id: int, tender_url: str):
    setup_worker_logging(run_id)
    run_dir = ELIGIBILITY_WORKER_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    await update_db_status(tender_pk_id, "processing")
    update_status(run_dir, "processing", "Worker started, preparing browser...")
    
    async with async_playwright() as p:
        browser, context, page = None, None, None
        download_path = None
        try:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()

            domain = urljoin(tender_url, '/')
            logger.info(f"Navigating to homepage '{domain}' to establish session.")
            await page.goto(domain, wait_until="domcontentloaded", timeout=60000)
            await handle_popups_and_timeouts(page)

            logger.info("Navigating to tender detail page...")
            await page.goto(tender_url, wait_until="domcontentloaded", timeout=60000)
            await handle_popups_and_timeouts(page)

            logger.info("Looking for 'Download as Zip' link...")
            zip_download_link = page.locator("a:has(img[src*='zipicon']), a:has-text('Download as zip file')").first
            await zip_download_link.wait_for(state="visible", timeout=15000)
            
            relative_zip_link_href = await zip_download_link.get_attribute('href')
            full_captcha_page_url = urljoin(tender_url, relative_zip_link_href)
            
            logger.info(f"Found zip link. Navigating to CAPTCHA page: {full_captcha_page_url}")
            await page.goto(full_captcha_page_url, wait_until="domcontentloaded", timeout=60000)
            await handle_popups_and_timeouts(page)

            captcha_image_element = page.locator("#captchaImage")
            # --- FIX: Check for the captcha image to verify we are on the correct page ---
            if not await captcha_image_element.is_visible(timeout=10000):
                raise Exception(f"Landed on page '{page.url}' but it does not contain a CAPTCHA image.")

            logger.info("On CAPTCHA page. Saving CAPTCHA for user.")
            await captcha_image_element.wait_for(state="visible", timeout=30000)
            captcha_b64_bytes = await captcha_image_element.screenshot(type='png')
            (run_dir / 'captcha.b64').write_text(base64.b64encode(captcha_b64_bytes).decode('utf-8'))
            update_status(run_dir, "WAITING_CAPTCHA")

            captcha_solution = await wait_for_captcha_solution(run_dir)
            if not captcha_solution: raise Exception("Failed to get CAPTCHA solution from user.")

            update_status(run_dir, "processing", "CAPTCHA received. Attempting to download...")
            await page.locator("#captchaText").fill(captcha_solution)
            
            try:
                async with page.expect_download(timeout=90000) as download_info:
                    await page.locator("#Submit").click()
                download = await download_info.value
                download_path = run_dir / download.suggested_filename
                await download.save_as(download_path)
                logger.info(f"✅ Download started directly. Saved to: {download_path}")
            except PlaywrightTimeout:
                logger.warning("Download did not start after CAPTCHA submit. Assuming redirect. Re-clicking download link.")
                await handle_popups_and_timeouts(page)
                zip_download_link_retry = page.locator("a:has(img[src*='zipicon']), a:has-text('Download as zip file')").first
                async with page.expect_download(timeout=60000) as download_info_retry:
                    await zip_download_link_retry.click()
                download = await download_info_retry.value
                download_path = run_dir / download.suggested_filename
                await download.save_as(download_path)
                logger.info(f"✅ Download successful on second attempt. Saved to: {download_path}")

        except Exception as e:
            logger.error(f"An error occurred during download phase: {e}", exc_info=True)
            # --- FIX: Use the correct variable name `run_dir` ---
            if page and not page.is_closed():
                await page.screenshot(path=run_dir / "error_screenshot.png")
                logger.error(f"Saved error screenshot to: {run_dir / 'error_screenshot.png'}")
            await update_db_status(tender_pk_id, "failed", {"error": str(e)})
            if context: await context.close()
            if browser: await browser.close()
            return
        
        # --- Post-download processing ---
        try:
            update_status(run_dir, "processing", "Download complete. Extracting files...")
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(run_dir)
            logger.info(f"✅ Unzipped files to: {run_dir}")
            
            primary_doc_path = find_primary_document(run_dir)
            if not primary_doc_path: raise Exception("Could not identify a primary PDF document to process.")
            
            extracted_text = extract_text_from_pdf(primary_doc_path)
            if not extracted_text: raise Exception(f"Failed to extract text from {primary_doc_path.name}")
            
            update_status(run_dir, "processing", "Document text extracted. Analyzing with AI...")
            db = SessionLocal()
            company_profile = db.query(CompanyProfile).first()
            db.close()
            if not company_profile: raise Exception("Company Profile not found in database.")
            
            prompt = f"""
            Analyze the following tender document text to determine if my company is eligible.

            My Company Profile:
            {json.dumps(company_profile.profile_data, indent=2)}

            ---
            Tender Document Extracted Text (first 12000 characters):
            {extracted_text[:12000]} 
            ---

            Please provide a structured JSON response with two keys:
            1. "clause_analysis": A list of objects, where each object has "clause", "requirement", "company_profile_match", and "conclusion" ('Met', 'Not Met', or 'Needs Manual Review').
            2. "final_summary": A brief, one-paragraph summary of the overall eligibility.
            """
            
            logger.info("Sending prompt to AI proxy...")
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{DASHBOARD_URL}/ai/proxy-chat", json={"prompt": prompt}, timeout=180.0)
                response.raise_for_status()
                ai_response = response.json()
            
            ai_content_str = ai_response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            try: analysis_result = json.loads(ai_content_str)
            except json.JSONDecodeError: analysis_result = {"raw_response": ai_content_str}

            score = -1
            if isinstance(analysis_result, dict) and "clause_analysis" in analysis_result:
                clauses = analysis_result["clause_analysis"]
                if isinstance(clauses, list) and len(clauses) > 0:
                    met_count = sum(1 for item in clauses if isinstance(item, dict) and str(item.get("conclusion")).strip().lower() == "met")
                    score = int((met_count / len(clauses)) * 100)
            
            logger.info(f"✅ AI analysis complete. Calculated score: {score}. Saving result to database.")
            update_status(run_dir, "complete", "Analysis complete!")
            await update_db_status(tender_pk_id, "complete", analysis_result, score)

        except Exception as e:
             logger.error(f"An error occurred during post-download processing: {e}", exc_info=True)
             update_status(run_dir, "failed", str(e))
             await update_db_status(tender_pk_id, "failed", {"error": str(e)})
        finally:
            # --- FIX: Correct cleanup logic ---
            if context:
                try: await context.close()
                except Exception: pass
            if browser and browser.is_connected():
                try: await browser.close()
                except Exception: pass
            logger.info("Eligibility check worker finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eligibility Check Worker for a single tender.")
    parser.add_argument("--run_id", required=True, help="Unique ID for this worker run.")
    parser.add_argument("--tender_pk_id", required=True, type=int, help="Database Primary Key of the tender.")
    parser.add_argument("--tender_url", required=True, help="Direct URL to the tender detail page.")
    args = parser.parse_args()
    
    sys.path.append(str(Path(__file__).parent.resolve()))
    asyncio.run(run_eligibility_check(run_id=args.run_id, tender_pk_id=args.tender_pk_id, tender_url=args.tender_url))
