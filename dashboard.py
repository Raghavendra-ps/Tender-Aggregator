# Tender-Aggregator-main/dashboard.py

import os
import re
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
import datetime
import logging
import subprocess
import sys
from urllib.parse import urljoin, urlparse

from fastapi import FastAPI, Request, Form, HTTPException, status, Depends, Response, APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
import httpx
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, or_

from database import SessionLocal, Tender, TenderResult, CompanyProfile, EligibilityCheck, Bidder, CanonicalBidder, User, init_db
from auth import create_access_token, require_user, require_admin, verify_password
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
ELIGIBILITY_WORKER_RUNS_DIR = PROJECT_ROOT / "TEMP" / "EligibilityRuns"
ELIGIBILITY_WORKER_RUNS_DIR.mkdir(parents=True, exist_ok=True)
SCRAPE_LOCK_FILE = PROJECT_ROOT / "scrape_in_progress.lock"
TEMP_WORKER_RUNS_DIR = PROJECT_ROOT / "TEMP" / "WorkerRuns"

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

app = FastAPI(title="TenFin Tender Dashboard")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- Jinja2 Filters ---
def neat_format(text: Optional[str]) -> str:
    if not text or not isinstance(text, str): return ""
    return text.replace('_', ' ').title()
def format_datetime_filter(value, format='%Y-%m-%d %H:%M'):
    if isinstance(value, datetime.datetime): return value.strftime(format)
    return value
templates.env.filters['neat_format'] = neat_format
templates.env.filters['datetime'] = format_datetime_filter

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@app.on_event("startup")
def on_startup():
    init_db()
    if SCRAPE_LOCK_FILE.exists():
        logger.warning("Stale scrape lock file found on startup. Removing it.")
        SCRAPE_LOCK_FILE.unlink()

def _clean_path_component(component: str) -> str:
    if not isinstance(component, str): return ""
    return re.sub(r'[^\w\-._]', '', component)
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

# --- Authentication Endpoints (PUBLIC) ---
@app.get("/login", response_class=HTMLResponse, name="login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": request.query_params.get("error")})

@app.post("/token", name="login_for_access_token")
async def login_for_access_token(response: Response, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password) or not user.is_active:
        return RedirectResponse(url="/login?error=1", status_code=status.HTTP_303_SEE_OTHER)
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True, samesite="lax")
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/logout", name="logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

# --- Protected API Routes ---
eligibility_router = APIRouter(prefix="/eligibility", tags=["eligibility"], dependencies=[Depends(require_user)])
class EligibilityRequest(BaseModel): tender_url: str

@eligibility_router.get("/get-status/{run_id}")
async def get_eligibility_worker_status(run_id: str):
    clean_run_id = _clean_path_component(run_id)
    run_dir = ELIGIBILITY_WORKER_RUNS_DIR / clean_run_id
    if not run_dir.is_dir(): raise HTTPException(status_code=404, detail="Run ID not found.")
    status_file = run_dir / "status.json"
    if not status_file.is_file(): return {"status": "starting", "message": "Worker is starting..."}
    try:
        status_data = json.loads(status_file.read_text(encoding='utf-8'))
        if status_data.get("status") == "WAITING_CAPTCHA":
            b64_file = run_dir / "captcha.b64"
            if b64_file.is_file():
                status_data["image_data"] = f"data:image/png;base64,{b64_file.read_text(encoding='utf-8')}"
        return status_data
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@eligibility_router.post("/submit-answer/{run_id}")
async def submit_eligibility_captcha_answer(run_id: str, captcha_text: str = Form(...)):
    clean_run_id = _clean_path_component(run_id)
    run_dir = ELIGIBILITY_WORKER_RUNS_DIR / clean_run_id
    if not run_dir.is_dir(): raise HTTPException(status_code=404, detail="Run ID not found.")
    (run_dir / "answer.txt").write_text(captcha_text, encoding='utf-8')
    return {"message": "CAPTCHA answer submitted successfully."}

@eligibility_router.post("/start-check/{tender_pk_id}")
async def start_eligibility_check(tender_pk_id: int, request_body: EligibilityRequest):
    worker_script = PROJECT_ROOT / "eligibility_worker.py"
    if not worker_script.is_file(): raise HTTPException(status_code=500, detail="Eligibility worker script not found.")
    run_id = f"eligibility_{tender_pk_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    command = [sys.executable, str(worker_script), "--run_id", run_id, "--tender_pk_id", str(tender_pk_id), "--tender_url", request_body.tender_url]
    subprocess.Popen(command, cwd=PROJECT_ROOT)
    return {"message": f"Eligibility check initiated.", "run_id": run_id}

app.include_router(eligibility_router)

# --- PROTECTED UI ROUTES ---
@app.get("/", response_class=HTMLResponse, name="homepage")
async def homepage(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_user)):
    live_tenders = db.query(Tender).filter(Tender.status != 'Result Announced').order_by(desc(Tender.published_date)).limit(100).all()
    tender_results = db.query(TenderResult).join(Tender).order_by(desc(TenderResult.created_at)).limit(100).all()
    
    combined_list, processed_tender_ids = [], set()
    for tender in live_tenders:
        combined_list.append({"type": "live_tender", "date": tender.published_date, "data": tender})
        processed_tender_ids.add(tender.id)
    for result in tender_results:
        if result.tender.id not in processed_tender_ids:
            combined_list.append({"type": "tender_result", "date": result.created_at, "data": result})
    
    combined_list.sort(key=lambda x: x.get('date') or datetime.datetime.min, reverse=True)
    
    total_tenders = db.query(func.count(Tender.id)).scalar()
    
    response = templates.TemplateResponse("index_db.html", {
        "request": request,
        "combined_items": combined_list,
        "total_tenders": total_tenders,
        "current_user": current_user
    })
    
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/tender/{tender_pk_id}", response_class=HTMLResponse, name="view_tender_detail")
async def view_tender_detail(request: Request, tender_pk_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_user)):
    tender = db.query(Tender).filter(Tender.id == tender_pk_id).first()
    if not tender: raise HTTPException(404, "Tender not found")
    detail_link = (tender.full_details_json or {}).get('detail_page_link', '')
    return templates.TemplateResponse("tender_detail.html", {"request": request, "tender": tender, "full_details_json": tender.full_details_json or {}, "detail_page_link": detail_link, "tender_pk_id": tender_pk_id, "current_user": current_user})

@app.get("/result/{tender_pk_id}", response_class=HTMLResponse, name="view_tender_result")
async def view_tender_result(request: Request, tender_pk_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_user)):
    result = db.query(TenderResult).filter(TenderResult.tender_id_fk == tender_pk_id).first()
    if not result: raise HTTPException(404, "Result not found")
    return templates.TemplateResponse("rot_summary_detail_db.html", {"request": request, "tender": result.tender, "parsed_data": result.full_summary_json or {}, "current_user": current_user})

@app.get("/settings", name="settings_page", response_class=HTMLResponse)
async def settings_page(request: Request, current_user: User = Depends(require_admin)):
    return templates.TemplateResponse("settings.html", {"request": request, "settings": load_settings(), "current_user": current_user})

@app.get("/rot-manual-scrape", name="rot_manual_scrape_page", response_class=HTMLResponse)
async def rot_manual_scrape_page(request: Request, current_user: User = Depends(require_admin)):
    return templates.TemplateResponse("rot_manual_scrape.html", {"request": request, "sites_for_rot": load_settings().get('site_configurations', {}), "current_user": current_user})

@app.get("/view-logs", name="view_logs_page", response_class=HTMLResponse)
async def view_logs_page(request: Request, site_log: Optional[str] = None, log_type: str = "regular", current_user: User = Depends(require_admin)):
    def get_lines(p: Path, n=200) -> str:
        if not p.is_file(): return f"[Log not found: {p.name}]"
        try:
            with open(p, 'rb') as f:
                f.seek(0, 2); end = f.tell(); lines_to_go=n+1; size=1024; read=0; blocks=[]
                while lines_to_go>0 and read<end:
                    s = min(size, end-read); f.seek(-(read+s), 2); blocks.append(f.read(s)); read+=s; lines_to_go-=blocks[-1].count(b'\n')
            return "\n".join(b''.join(reversed(blocks)).decode('utf-8','replace').splitlines()[-n:])
        except: return f"[Error reading log: {p.name}]"
    
    regular_dir = LOGS_BASE_DIR / "regular_scraper"
    rot_dir = LOGS_BASE_DIR / "rot_worker"
    reg_logs = sorted([p.name for p in regular_dir.glob("scrape_*.log")]) if regular_dir.is_dir() else []
    rot_logs = sorted([f"{d.name}/{f.name}" for d in rot_dir.iterdir() if d.is_dir() for f in d.glob("worker*.log")]) if rot_dir.is_dir() else []
    
    content = None
    if site_log:
        log_path = (rot_dir / site_log) if log_type == "rot" else (regular_dir / site_log)
        content = get_lines(log_path)

    return templates.TemplateResponse("view_logs.html", {
        "request": request,
        "controller_log_filename": "site_controller.log",
        "controller_log_content": get_lines(LOGS_BASE_DIR / "site_controller.log"),
        "regular_site_log_files": reg_logs,
        "rot_site_log_files": rot_logs,
        "selected_site_log_filename": site_log,
        "selected_site_log_content": content,
        "current_log_type_filter": log_type,
        "current_user": current_user
    })

@app.get("/download-log/{log_type}/{path_param:path}", name="download_log_typed")
async def download_log_typed(log_type: str, path_param: str, current_user: User = Depends(require_admin)):
    log_dir_map = {
        "controller": LOGS_BASE_DIR,
        "regular": LOGS_BASE_DIR / "regular_scraper",
        "rot": LOGS_BASE_DIR / "rot_worker"
    }
    base_dir = log_dir_map.get(log_type)
    if not base_dir: raise HTTPException(status_code=400, detail="Invalid log type specified.")
    
    log_path = (base_dir / path_param).resolve()
    if not str(log_path).startswith(str(base_dir.resolve())):
        raise HTTPException(status_code=403, detail="Access forbidden.")
    if not log_path.is_file():
        raise HTTPException(status_code=404, detail="Log file not found.")
        
    return FileResponse(path=log_path, filename=log_path.name, media_type='text/plain')


# --- ACTION & POST ROUTES (ADMIN) ---

@app.post("/run-scraper-now", name="run_scraper_now")
async def run_scraper_now(current_user: User = Depends(require_admin)):
    if SCRAPE_LOCK_FILE.exists():
        return JSONResponse(status_code=409, content={"status": "error", "message": "A scrape is already in progress."})
    subprocess.Popen([sys.executable, str(SITE_CONTROLLER_SCRIPT_PATH), "--type", "regular"], cwd=PROJECT_ROOT)
    return JSONResponse(status_code=202, content={"status": "ok", "message": "Scrape initiated."})

@app.get("/get-scrape-status", name="get_scrape_status")
async def get_scrape_status(current_user: User = Depends(require_user)):
    return {"status": "running" if SCRAPE_LOCK_FILE.exists() else "idle"}

@app.post("/save-scraper-parameters", name="save_scraper_parameters")
async def save_scraper_parameters(request: Request, current_user: User = Depends(require_admin)):
    form = await request.form(); s = load_settings(); gss = s.setdefault("global_scraper_settings", {})
    gss.setdefault("concurrency", {})['list_pages'] = int(form.get('concurrency_list', 5))
    gss["concurrency"]['detail_pages'] = int(form.get('concurrency_detail', 2))
    # Add other parameters from form as needed
    if save_settings(s):
        return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=scraper_params_saved", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url=app.url_path_for('settings_page') + "?error=scraper_params_save_failed", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/save-enabled-sites", name="save_enabled_sites")
async def save_enabled_sites(request: Request, current_user: User = Depends(require_admin)):
    form = await request.form(); settings = load_settings()
    for site in settings.get("site_configurations", {}):
        settings["site_configurations"][site]["enabled"] = form.get(f"site_{site}") == "on"
    if save_settings(settings):
        return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=enabled_sites_saved", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url=app.url_path_for('settings_page') + "?error=enabled_sites_save_failed", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/save-main-schedule", name="save_main_schedule")
async def save_main_schedule(request: Request, current_user: User = Depends(require_admin)):
    form=await request.form(); settings=load_settings(); sched=settings.setdefault("scheduler", {})
    sched["main_enabled"] = form.get("enable_main_schedule") == "True"
    for k in ["main_frequency", "main_time", "main_day_weekly", "main_day_monthly"]:
        if k in form: sched[k]=form[k]
    if save_settings(settings):
        return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=schedule_settings_saved", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url=app.url_path_for('settings_page') + "?error=schedule_settings_save_failed", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/apply-schedule", name="apply_schedule")
async def apply_schedule(current_user: User = Depends(require_admin)):
    res=subprocess.run([sys.executable, str(SCHEDULER_SETUP_SCRIPT_PATH)], cwd=PROJECT_ROOT)
    if res.returncode==0:
        return RedirectResponse(url=app.url_path_for('settings_page') + "?msg=schedule_applied_successfully", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(url=app.url_path_for('settings_page') + "?error=schedule_apply_failed", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/rot/initiate-rot-worker/{site_key}", name="initiate_rot_worker")
async def initiate_rot_worker_endpoint(request: Request, site_key: str, tender_status_value: str = Form(...), from_date: Optional[str] = Form(None), to_date: Optional[str] = Form(None), current_user: User = Depends(require_admin)):
    worker_script_path=PROJECT_ROOT/"headless_rot_worker.py"
    if not worker_script_path.is_file(): raise HTTPException(500, "ROT worker script missing.")
    run_id=f"rot_worker_{_clean_path_component(site_key)}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    run_dir=TEMP_WORKER_RUNS_DIR/run_id; run_dir.mkdir(parents=True, exist_ok=True)
    rot_urls=get_rot_site_config(site_key)
    if not rot_urls: raise HTTPException(500, f"Config error for ROT site '{site_key}'.")
    settings=load_settings(); gss=settings.get("global_scraper_settings",{}); worker_settings={**gss.get("rot_timeouts",{}), **gss.get("rot_scrape_limits",{}), **gss.get("rot_concurrency",{})}
    (run_dir/"status.txt").write_text("INITIATED_BY_DASHBOARD", encoding="utf-8")
    command=[sys.executable, str(worker_script_path), "--site_key", site_key, "--run_id", run_id, "--site_url", rot_urls["main_search_url"], "--base_url", rot_urls["base_url"], "--tender_status", tender_status_value, "--settings_json", json.dumps(worker_settings)]
    if from_date: command.extend(["--from_date", from_date])
    if to_date: command.extend(["--to_date", to_date])
    subprocess.Popen(command, cwd=PROJECT_ROOT)
    return JSONResponse(status_code=202, content={"message": "ROT Worker initiated.", "run_id": run_id})

@app.get("/get-captcha-status/{run_id}", name="get_captcha_status")
async def get_captcha_status_route(run_id: str, current_user: User = Depends(require_admin)):
    run_dir = TEMP_WORKER_RUNS_DIR / _clean_path_component(run_id)
    if not run_dir.is_dir(): return JSONResponse(status_code=404, content={"status": "ERROR_RUN_DIR_NOT_FOUND"})
    status_file = run_dir / "status.txt"
    status_msg = status_file.read_text().strip() if status_file.is_file() else "pending"
    image_data = None
    if status_msg == "WAITING_CAPTCHA":
        b64_file = run_dir / "captcha.b64"
        if b64_file.is_file(): image_data = f"data:image/png;base64,{b64_file.read_text().strip()}"
    return JSONResponse({"status": status_msg, "image_data": image_data})

@app.post("/submit-captcha-answer/{run_id}", name="submit_captcha_answer")
async def submit_captcha_answer_route(run_id: str, captcha_text: str = Form(...), current_user: User = Depends(require_admin)):
    run_dir = TEMP_WORKER_RUNS_DIR / _clean_path_component(run_id)
    if not run_dir.is_dir(): return JSONResponse(status_code=404)
    (run_dir / "answer.txt").write_text(captcha_text)
    (run_dir / "status.txt").write_text("CAPTCHA_ANSWER_SUBMITTED")
    return JSONResponse({"message": "CAPTCHA answer submitted."})


# --- Company Profile & Competitor Management ---

@app.get("/company-profile", name="company_profile_page", response_class=HTMLResponse)
async def company_profile_page(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_user)):
    profile_record = db.query(CompanyProfile).filter(CompanyProfile.profile_name == "Default Profile").first()
    return templates.TemplateResponse("company_profile.html", {"request": request, "profile": profile_record.profile_data if profile_record else {}, "msg": request.query_params.get('msg'), "current_user": current_user})

@app.post("/company-profile", name="save_company_profile")
async def save_company_profile(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_user)):
    form_data = await request.form()
    profile_dict = {key: value for key, value in form_data.items()}
    profile_record = db.query(CompanyProfile).filter(CompanyProfile.profile_name == "Default Profile").first()
    if profile_record:
        profile_record.profile_data = profile_dict
    else:
        profile_record = CompanyProfile(profile_name="Default Profile", profile_data=profile_dict)
        db.add(profile_record)
    db.commit()
    return RedirectResponse(url=app.url_path_for('company_profile_page') + "?msg=Profile+saved", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/competitors", name="competitor_management_page", response_class=HTMLResponse)
async def competitor_management_page(request: Request, db: Session = Depends(get_db), current_user: User = Depends(require_admin)):
    bidders = db.query(Bidder).filter(Bidder.canonical_id == None).order_by(Bidder.bidder_name).all()
    return templates.TemplateResponse("competitor_management.html", {"request": request, "bidders": bidders, "msg": request.query_params.get('msg'), "current_user": current_user})

@app.post("/competitors/merge", name="merge_bidders")
async def merge_bidders(primary_bidder_id: int = Form(...), alias_ids: list[int] = Form(...), db: Session = Depends(get_db), current_user: User = Depends(require_admin)):
    primary_bidder = db.query(Bidder).filter(Bidder.id == primary_bidder_id).first()
    if not primary_bidder: raise HTTPException(status_code=404, detail="Primary bidder not found.")
    canonical_bidder = db.query(CanonicalBidder).filter(CanonicalBidder.canonical_name == primary_bidder.bidder_name).first()
    if not canonical_bidder:
        canonical_bidder = CanonicalBidder(canonical_name=primary_bidder.bidder_name)
        db.add(canonical_bidder)
        db.flush()
    all_ids_to_update = alias_ids + [primary_bidder_id]
    bidders_to_merge = db.query(Bidder).filter(Bidder.id.in_(all_ids_to_update)).all()
    for bidder in bidders_to_merge:
        bidder.canonical_id = canonical_bidder.id
    db.commit()
    return RedirectResponse(url=app.url_path_for('competitor_management_page') + "?msg=Bidders+merged+successfully!", status_code=status.HTTP_303_SEE_OTHER)

# --- AI Proxy Route ---
class AIPrompt(BaseModel):
    prompt: str

@app.post("/ai/proxy-chat", name="proxy_ai_chat")
async def proxy_ai_chat(prompt_data: AIPrompt, current_user: User = Depends(require_user)):
    AI_API_KEY = os.environ.get("AI_API_KEY", "YOUR_AI_API_KEY_HERE")
    AI_API_URL = "https://api.openai.com/v1/chat/completions"

    if not AI_API_KEY or AI_API_KEY == "YOUR_AI_API_KEY_HERE":
        logger.error("AI_API_KEY is not configured on the server.")
        return JSONResponse(
            status_code=200,
            content={"choices": [{"message": {"role": "assistant", "content": "SIMULATED AI RESPONSE: API Key not found."}}]}
        )

    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "gpt-3.5-turbo", "messages": [{"role": "system", "content": "You are an expert assistant."}, {"role": "user", "content": prompt_data.prompt}]}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(AI_API_URL, json=payload, headers=headers, timeout=120.0)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"AI service error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during AI proxy request: {e}")
