# Tender-Aggregator-main/dashboard.py

import os
import re
import io
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime
import logging
import subprocess
import sys
from urllib.parse import urlparse, urljoin

from fastapi import FastAPI, Request, Form, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel
import httpx

# --- Database Imports ---
from database import SessionLocal, Tender, TenderResult, init_db
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, or_

# --- Local Module Imports ---
try:
    from headless_rot_worker import URL_SUFFIX_ROT
except ImportError:
    URL_SUFFIX_ROT = "?page=WebTenderStatusLists&service=page"

# --- Path and Config Constants ---
PROJECT_ROOT = Path(__file__).parent.resolve()
TEMPLATES_DIR = PROJECT_ROOT / "templates"
SETTINGS_FILE = PROJECT_ROOT / "settings.json"
LOGS_BASE_DIR = PROJECT_ROOT / "LOGS"
SITE_CONTROLLER_SCRIPT_PATH = PROJECT_ROOT / "site_controller.py"
SCHEDULER_SETUP_SCRIPT_PATH = PROJECT_ROOT / "scheduler_setup.py"
SITE_DATA_ROOT = PROJECT_ROOT / "site_data"
ROT_DETAIL_HTMLS_DIR = SITE_DATA_ROOT / "ROT" / "DetailHtmls"
TEMP_DATA_DIR = SITE_DATA_ROOT / "TEMP"
TEMP_WORKER_RUNS_DIR = TEMP_DATA_DIR / "WorkerRuns"

# --- Logging Setup ---
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] (Dashboard) %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("dashboard")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(LOGS_BASE_DIR / "dashboard.log", mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.propagate = False

# --- DEFAULT_SETTINGS ---
DEFAULT_SETTINGS = { "global_scraper_settings": {"scrape_py_limits": {}, "concurrency": {}, "timeouts": {}, "rot_scrape_limits": {}, "rot_concurrency": {}, "rot_timeouts": {}}, "retention": {}, "scheduler": {}, "site_configurations": {} }

# --- FastAPI App & Templates ---
app = FastAPI(title="TenFin Tender Dashboard")
templates: Optional[Jinja2Templates] = None
try:
    jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(['html', 'xml']))
    def format_datetime(value, format='%Y-%m-%d %H:%M'):
        if isinstance(value, datetime.datetime): return value.strftime(format)
        return value
    def neat_format(text: Optional[str]) -> str:
        if not text or not isinstance(text, str): return ""
        return text.replace('_', ' ').title()
    jinja_env.filters['datetime'] = format_datetime
    jinja_env.filters['neat_format'] = neat_format
    templates = Jinja2Templates(env=jinja_env)
except Exception as e:
    logger.error(f"Jinja2 init failed: {e}")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@app.on_event("startup")
def on_startup():
    init_db()

# --- Helper Functions ---
def load_settings() -> Dict:
    if not SETTINGS_FILE.is_file(): return DEFAULT_SETTINGS
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except: return DEFAULT_SETTINGS

def save_settings(settings_data: Dict) -> bool:
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings_data, f, indent=2)
        return True
    except: return False

def _clean_path_component(component: str) -> str:
    if not isinstance(component, str): return ""
    cleaned = re.sub(r'[^\w\-._]', '', component)
    if cleaned in [".", ".."]: return ""
    return cleaned

def get_rot_site_config(site_key: str) -> Optional[Dict[str, str]]:
    settings = load_settings()
    config = settings.get("site_configurations", {}).get(site_key)
    if config and config.get("domain"):
        domain = config["domain"].rstrip('/?')
        search_url = f"{domain}{URL_SUFFIX_ROT}"
        parsed = urlparse(domain)
        path = parsed.path if parsed.path.endswith('/') else f"{parsed.path}/"
        base_url = f"{parsed.scheme}://{parsed.netloc}{path}"
        if 'nicgep' in base_url.lower() and not base_url.endswith('app/'):
            base_url = urljoin(base_url, 'app/')
        return {"main_search_url": search_url, "base_url": base_url}
    return None

# === MAIN UI & DATA ROUTES ===

@app.get("/", response_class=HTMLResponse, name="homepage")
async def homepage(request: Request, db: Session = Depends(get_db), keywords: Optional[str] = None, site_key: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None):
    if not templates: return HTMLResponse("Template engine error", 503)
    try:
        query = db.query(Tender)
        is_search = any([keywords, site_key, start_date, end_date])
        if is_search:
            if keywords:
                search_clauses = [or_(Tender.tender_title.ilike(f"%{kw}%"), Tender.organisation_chain.ilike(f"%{kw}%"), Tender.full_details_json.op('->>')('$').ilike(f"%{kw}%")) for kw in [k.strip() for k in keywords.split(',') if k.strip()]]
                if search_clauses: query = query.filter(or_(*search_clauses))
            if site_key: query = query.filter(Tender.source_site == site_key)
            if start_date: query = query.filter(Tender.opening_date >= start_date)
            if end_date: query = query.filter(Tender.opening_date < (datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1)))
        tenders = query.order_by(desc(Tender.published_date)).limit(200).all()
        total_tenders = db.query(func.count(Tender.id)).scalar()
        sites_in_db = [s[0] for s in db.query(Tender.source_site).distinct().order_by(Tender.source_site).all() if s[0]]
    except Exception as e:
        logger.error(f"DB Error on homepage: {e}", exc_info=True)
        return templates.TemplateResponse("error.html", {"request": request, "error": "Database query failed."})
    return templates.TemplateResponse("index_db.html", {"request": request, "tenders": tenders, "total_tenders": total_tenders, "tenders_shown": len(tenders), "available_sites": sites_in_db, "is_search": is_search, "query_keywords": keywords, "query_site": site_key, "query_start_date": start_date, "query_end_date": end_date})

@app.get("/tender/{tender_pk_id}", name="view_tender_detail", response_class=HTMLResponse)
async def view_tender_detail(request: Request, tender_pk_id: int, db: Session = Depends(get_db)):
    if not templates: raise HTTPException(503)
    tender = db.query(Tender).filter(Tender.id == tender_pk_id).first()
    if not tender: raise HTTPException(404)
    return templates.TemplateResponse("tender_detail.html", {"request": request, "tender": tender.full_details_json or {}})

@app.get("/result/{tender_pk_id}", name="view_tender_result", response_class=HTMLResponse)
async def view_tender_result(request: Request, tender_pk_id: int, db: Session = Depends(get_db)):
    if not templates: raise HTTPException(503)
    result = db.query(TenderResult).filter(TenderResult.tender_id_fk == tender_pk_id).first()
    if not result: raise HTTPException(404)
    return templates.TemplateResponse("rot_summary_detail_db.html", {"request": request, "tender": result.tender, "parsed_data": result.full_summary_json or {}})

@app.get("/settings", name="settings_page", response_class=HTMLResponse)
async def settings_page(request: Request):
    if not templates: return HTMLResponse("Template engine error", 503)
    return templates.TemplateResponse("settings.html", {"request": request, "settings": load_settings()})

@app.get("/rot-manual-scrape", name="rot_manual_scrape_page", response_class=HTMLResponse)
async def rot_manual_scrape_page(request: Request):
    if not templates: raise HTTPException(503)
    return templates.TemplateResponse("rot_manual_scrape.html", {"request": request, "sites_for_rot": load_settings().get('site_configurations', {}), "settings": load_settings()})

@app.get("/view-logs", name="view_logs_page", response_class=HTMLResponse)
async def view_logs_page(request: Request, site_log: Optional[str] = None, log_type: str = "regular"):
    if not templates: raise HTTPException(503)
    log_lines=200; controller_log_path = LOGS_BASE_DIR/"site_controller.log"
    def get_lines(p: Path) -> str:
        if not p.is_file(): return f"[Log not found: {p.name}]"
        try:
            with open(p, 'rb') as f:
                f.seek(0, 2); end = f.tell(); lines_to_go=log_lines+1; size=1024; read=0; blocks=[]
                while lines_to_go>0 and read<end:
                    s = min(size, end-read); f.seek(-(read+s), 2); blocks.append(f.read(s)); read+=s; lines_to_go-=blocks[-1].count(b'\n')
            return "\n".join(b''.join(reversed(blocks)).decode('utf-8','replace').splitlines()[-log_lines:])
        except: return f"[Error reading log: {p.name}]"
    regular_dir, rot_dir = LOGS_BASE_DIR/"regular_scraper", LOGS_BASE_DIR/"rot_worker"
    reg_logs = sorted([p.name for p in regular_dir.glob("scrape_*.log")]) if regular_dir.is_dir() else []
    rot_logs = sorted([f"{d.name}/{f.name}" for d in rot_dir.iterdir() if d.is_dir() for f in d.glob("worker*.log")]) if rot_dir.is_dir() else []
    content = get_lines((rot_dir / site_log) if log_type == "rot" else (regular_dir / site_log)) if site_log else None
    return templates.TemplateResponse("view_logs.html", {"request":request, "controller_log_filename":"site_controller.log", "controller_log_content":get_lines(controller_log_path), "regular_site_log_files":reg_logs, "rot_site_log_files":rot_logs, "selected_site_log_filename":site_log, "selected_site_log_content":content, "current_log_type_filter":log_type})

#@app.get("/download-log/{log_type}/{path_param:path}", name="download_log_typed")
#async def download_log_typed(log_type: str, path_param: str):
    #log_dir_map = {"controller": LOGS_BASE_DIR, "regular": LOGS_BASE_DIR/"regular_scraper", "rot": LOGS_BASE_DIR/"rot_worker"}
    #base_dir = log_dir_map.get(log_type)
   # if not base_dir: raise HTTPException(400)
  #  log_path = (base_dir/path_param).resolve()
 #   if not str(log_path).startswith(str(base_dir.resolve())) or not log_path.is_file(): raise HTTPException(404)
#    return FileResponse(path=log_path, filename=log_path.name)

@app.get("/download-log/{log_type}/{path_param:path}", name="download_log_typed")
async def download_log_typed(log_type: str, path_param: str):
    log_dir_map = {
        "controller": LOGS_BASE_DIR,
        "regular": LOGS_BASE_DIR / "regular_scraper",
        "rot": LOGS_BASE_DIR / "rot_worker"
    }
    base_dir = log_dir_map.get(log_type)

    if not base_dir:
        raise HTTPException(status_code=400, detail="Invalid log type specified.")

    log_path = (base_dir / path_param).resolve()

    # Security check to prevent path traversal
    if not str(log_path).startswith(str(base_dir.resolve())):
        raise HTTPException(status_code=403, detail="Access to this path is forbidden.")
    
    if not log_path.is_file():
        raise HTTPException(status_code=404, detail=f"Log file not found: {path_param}")

    return FileResponse(path=log_path, filename=log_path.name, media_type='text/plain')

# --- ACTION & POST ROUTES ---
@app.post("/run-scraper-now", name="run_scraper_now")
async def run_scraper_now(request: Request):
    if not SITE_CONTROLLER_SCRIPT_PATH.is_file(): return RedirectResponse(app.url_path_for('homepage') + "?error=controller_script_missing", status_code=303)
    subprocess.Popen([sys.executable, str(SITE_CONTROLLER_SCRIPT_PATH), "--type", "regular"], cwd=PROJECT_ROOT)
    return RedirectResponse(url=app.url_path_for('homepage') + "?msg=manual_scrape_triggered", status_code=303)

@app.post("/save-retention-settings", name="save_retention_settings")
async def save_retention_settings(enable_retention: bool = Form(False), retention_days: int = Form(30)):
    settings = load_settings(); settings.setdefault("retention", {})["enabled"] = enable_retention; settings["retention"]["days"] = retention_days
    if save_settings(settings): return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=retention_settings_saved", status_code=303)
    return RedirectResponse(url=app.url_path_for('settings_page') + "?error=retention_save_failed", status_code=303)

@app.post("/save-scraper-parameters", name="save_scraper_parameters")
async def save_scraper_parameters(request: Request):
    form = await request.form(); s = load_settings(); gss = s.setdefault("global_scraper_settings", {})
    gss.setdefault("concurrency", {})['list_pages'] = int(form.get('concurrency_list', 5))
    gss["concurrency"]['detail_pages'] = int(form.get('concurrency_detail', 2))
    gss.setdefault("scrape_py_limits", {})['max_list_pages'] = int(form.get('max_list_pages', 100))
    gss["scrape_py_limits"]['retries'] = int(form.get('retries', 3))
    gss.setdefault("timeouts", {})['page_load'] = int(form.get('page_load_timeout', 75000))
    gss["timeouts"]['detail_page'] = int(form.get('detail_page_timeout', 75000))
    gss.setdefault("rot_scrape_limits", {})['max_list_pages'] = int(form.get('rot_max_list_pages', 50))
    gss.setdefault("rot_concurrency", {})['detail_processing'] = int(form.get('rot_detail_processing', 2))
    rot_timeouts = gss.setdefault("rot_timeouts", {})
    rot_timeouts['page_load'] = int(form.get('rot_page_load_timeout', 75000))
    rot_timeouts['detail_page'] = int(form.get('rot_detail_page_timeout', 60000))
    s.setdefault("scheduler", {})['rot_default_status_value'] = form.get('rot_default_status_value', '5')
    if save_settings(s): return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=scraper_params_saved", status_code=303)
    return RedirectResponse(url=app.url_path_for('settings_page') + "?error=scraper_params_save_failed", status_code=303)

@app.post("/save-enabled-sites", name="save_enabled_sites")
async def save_enabled_sites(request: Request):
    form = await request.form(); settings = load_settings()
    for site in settings.get("site_configurations", {}): settings["site_configurations"][site]["enabled"] = form.get(f"site_{site}") == "on"
    if save_settings(settings): return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=enabled_sites_saved", status_code=303)
    return RedirectResponse(url=app.url_path_for('settings_page') + "?error=enabled_sites_save_failed", status_code=303)

@app.post("/save-main-schedule", name="save_main_schedule")
async def save_main_schedule(request: Request):
    form=await request.form(); settings=load_settings(); sched=settings.setdefault("scheduler", {})
    sched["main_enabled"]=form.get("enable_main_schedule")=="True"
    for k in ["main_frequency","main_time","main_day_weekly","main_day_monthly"]:
        if k in form: sched[k]=form[k]
    if save_settings(settings): return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=schedule_settings_saved_apply_needed", status_code=303)
    return RedirectResponse(url=app.url_path_for('settings_page') + "?error=schedule_settings_save_failed", status_code=303)

@app.post("/apply-schedule", name="apply_schedule")
async def apply_schedule():
    if not SCHEDULER_SETUP_SCRIPT_PATH.is_file(): return RedirectResponse(app.url_path_for('settings_page') + "?error=scheduler_script_missing", status_code=303)
    res=subprocess.run([sys.executable, str(SCHEDULER_SETUP_SCRIPT_PATH)], cwd=PROJECT_ROOT)
    if res.returncode==0: return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=schedule_applied_successfully", status_code=303)
    return RedirectResponse(url=app.url_path_for('settings_page') + "?error=schedule_apply_failed", status_code=303)

@app.post("/run-cleanup", name="run_cleanup_task")
async def run_cleanup_task(db: Session = Depends(get_db)):
    settings=load_settings()
    if not settings.get("retention", {}).get("enabled", False): return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=cleanup_disabled", status_code=303)
    try:
        days=int(settings.get("retention",{}).get("days",30)); cutoff=datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(days=days)
        deleted=db.query(Tender).filter(Tender.created_at<cutoff).delete(); db.commit()
        return RedirectResponse(url=app.url_path_for('settings_page') + f"?msg=cleanup_finished_deleted_{deleted}", status_code=303)
    except Exception: db.rollback(); return RedirectResponse(app.url_path_for('settings_page') + "?error=cleanup_failed", status_code=303)

# === ROT WORKER & CAPTCHA ROUTES ===

@app.post("/rot/initiate-rot-worker/{site_key}", name="initiate_rot_worker")
async def initiate_rot_worker_endpoint(request: Request, site_key: str, tender_status_value: str = Form(...)):
    worker_script_path=PROJECT_ROOT/"headless_rot_worker.py"
    if not worker_script_path.is_file(): raise HTTPException(500, "ROT worker script missing.")
    run_id=f"rot_worker_{_clean_path_component(site_key)}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    run_dir=TEMP_WORKER_RUNS_DIR/run_id; run_dir.mkdir(parents=True, exist_ok=True)
    try:
        rot_urls=get_rot_site_config(site_key)
        if not rot_urls: raise HTTPException(500, f"Config error for ROT site '{site_key}'.")
        settings=load_settings(); gss=settings.get("global_scraper_settings",{}); worker_settings={**gss.get("rot_timeouts",{}), **gss.get("rot_scrape_limits",{}), **gss.get("rot_concurrency",{})}
        (run_dir/"status.txt").write_text("INITIATED_BY_DASHBOARD", encoding="utf-8")
        command=[sys.executable, str(worker_script_path), "--site_key", site_key, "--run_id", run_id, "--site_url", rot_urls["main_search_url"], "--base_url", rot_urls["base_url"], "--tender_status", tender_status_value, "--settings_json", json.dumps(worker_settings)]
        env=os.environ.copy(); env["TENFIN_DASHBOARD_HOST_PORT"]=f"{request.url.hostname}:{request.url.port}"
        subprocess.Popen(command, cwd=PROJECT_ROOT, env=env)
        logger.info(f"Launched ROT Worker for {site_key}. Run ID: {run_id}")
        return JSONResponse(status_code=202, content={"message": "ROT Worker initiated.", "run_id": run_id})
    except Exception as e:
        logger.error(f"Failed to launch worker for {site_key}: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to start ROT worker: {type(e).__name__}")

@app.get("/get-captcha-status/{run_id}", name="get_captcha_status")
async def get_captcha_status_route(run_id: str):
    run_dir = TEMP_WORKER_RUNS_DIR / _clean_path_component(run_id)
    if not run_dir.is_dir():
        logger.warning(f"[CAPTCHA STATUS] Run directory not found for run_id: {run_id}")
        return JSONResponse(status_code=404)

    status_file = run_dir / "status.txt"
    status_msg = status_file.read_text().strip() if status_file.is_file() else "pending"
    
    image_data = None
    if status_msg == "WAITING_CAPTCHA":
        b64_file = run_dir / "captcha.b64"
        if b64_file.is_file():
            image_data = f"data:image/png;base64,{b64_file.read_text().strip()}"
        else:
            logger.warning(f"[CAPTCHA STATUS] CAPTCHA image missing for run_id: {run_id} (status: WAITING_CAPTCHA)")
    
    logger.info(f"[CAPTCHA STATUS] Run ID: {run_id} | Status: {status_msg} | Captcha File Exists: {(run_dir / 'captcha.b64').is_file()}")
    return JSONResponse({"status": status_msg, "image_data": image_data})

@app.post("/submit-captcha-answer/{run_id}", name="submit_captcha_answer")
async def submit_captcha_answer_route(run_id: str, captcha_text: str = Form(...)):
    run_dir = TEMP_WORKER_RUNS_DIR / _clean_path_component(run_id)
    if not run_dir.is_dir(): return JSONResponse(status_code=404)
    (run_dir / "answer.txt").write_text(captcha_text)
    (run_dir / "status.txt").write_text("CAPTCHA_ANSWER_SUBMITTED")
    return JSONResponse({"message": "CAPTCHA answer submitted."})
