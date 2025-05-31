import sys
import json
from pathlib import Path
import getpass 
import shlex 
import datetime 
import os 
from typing import Optional, Dict

try:
    from crontab import CronTab, CronItem
except ImportError:
    print("[CRITICAL - SchedulerSetup] ERROR: python-crontab library not found. Please install it: pip install python-crontab")
    sys.exit(1)

# --- Configuration ---
try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
except NameError: 
    SCRIPT_DIR = Path('.').resolve()

SETTINGS_FILE = SCRIPT_DIR / "settings.json"
SITE_CONTROLLER_SCRIPT_PATH = SCRIPT_DIR / "site_controller.py"
PYTHON_EXEC = Path(sys.executable).resolve()

CONTROLLER_CRON_COMMENT_REGULAR = "TenFinScraperJob_SiteController_Regular"
# ROT_CONTROLLER_CRON_COMMENT = "TenFinScraperJob_SiteController_ROT" # REMOVED - ROT is manual
CLEANUP_CRON_COMMENT = "TenFinCleanupJob_DataRetention"

def log_message(message: str, level: str = "INFO"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp} - {level} - SchedulerSetup] {message}")

def load_settings() -> dict:
    if not SETTINGS_FILE.is_file():
        log_message(f"Settings file not found at {SETTINGS_FILE}", "ERROR")
        return {}
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        log_message(f"Successfully loaded settings from {SETTINGS_FILE}")
        # Remove ROT scheduler keys if they exist, as they are no longer used for cron
        if "scheduler" in settings:
            keys_to_remove = [
                "rot_enabled", "rot_frequency", "rot_time", 
                "rot_day_weekly", "rot_day_monthly", "rot_default_status_value"
            ]
            updated = False
            for key in keys_to_remove:
                if key in settings["scheduler"]:
                    del settings["scheduler"][key]
                    updated = True
            if updated:
                log_message("Removed obsolete ROT scheduler keys from loaded settings for this run.", "DEBUG")
                # Optionally, could re-save settings here, but dashboard should handle permanent removal on next save.
        return settings
    except Exception as e:
        log_message(f"Failed to load or parse settings from {SETTINGS_FILE}: {e}", "ERROR")
        return {}

def set_regular_controller_schedule(settings: dict, cron_instance: CronTab):
    """ Sets the schedule for the regular site_controller.py scrape. """
    
    schedule_config = settings.get("scheduler", {})
    is_enabled = schedule_config.get("main_enabled", False)
    frequency = schedule_config.get("main_frequency", "daily")
    run_time_str = schedule_config.get("main_time", "02:00")
    day_weekly = schedule_config.get("main_day_weekly", "7")
    day_monthly = schedule_config.get("main_day_monthly", "1")
    cron_comment_to_use = CONTROLLER_CRON_COMMENT_REGULAR
    log_prefix = "Regular Controller"
    controller_arg = "--type regular" 

    log_message(f"Processing {log_prefix} schedule: Enabled={is_enabled}")

    try:
        jobs_to_remove = list(cron_instance.find_comment(cron_comment_to_use))
        if jobs_to_remove:
            for job in jobs_to_remove:
                cron_instance.remove(job)
            log_message(f"Removed {len(jobs_to_remove)} existing cron jobs: {cron_comment_to_use}")
    except Exception as e_remove:
         log_message(f"Error removing existing {log_prefix} cron jobs ({cron_comment_to_use}): {e_remove}", "ERROR")

    if not is_enabled:
        log_message(f"{log_prefix} scheduling (main_enabled) is False. No cron job created.")
        return True 

    if not PYTHON_EXEC.is_file(): log_message(f"Python exec not found: {PYTHON_EXEC}", "ERROR"); return False
    if not SITE_CONTROLLER_SCRIPT_PATH.is_file(): log_message(f"Controller script not found: {SITE_CONTROLLER_SCRIPT_PATH}", "ERROR"); return False

    command = f'cd {shlex.quote(str(SCRIPT_DIR))} && {shlex.quote(str(PYTHON_EXEC))} {shlex.quote(str(SITE_CONTROLLER_SCRIPT_PATH))} {controller_arg}'
    log_message(f"Constructed {log_prefix} command: {command}")

    job: Optional[CronItem] = None
    try:
        job = cron_instance.new(command=command, comment=cron_comment_to_use)
    except Exception as e_new_job:
        log_message(f"Error creating new {log_prefix} cron job: {e_new_job}", "ERROR"); return False

    try: hour, minute = map(int, run_time_str.split(':'))
    except ValueError:
        log_message(f"Invalid time '{run_time_str}' for {log_prefix}. Using 02:00.", "WARN"); hour, minute = 2,0

    schedule_desc = ""; is_valid_schedule = False
    try:
        if frequency == "daily": job.setall(f'{minute} {hour} * * *'); schedule_desc = f"daily at {hour:02d}:{minute:02d}"
        elif frequency == "weekly":
            cron_day = int(day_weekly) % 7 
            job.setall(f'{minute} {hour} * * {cron_day}'); schedule_desc = f"weekly on day {day_weekly} (cron day {cron_day}) at {hour:02d}:{minute:02d}"
        elif frequency == "monthly":
            if day_monthly == "last": 
                job.setall(f'{minute} {hour} 28 * *'); schedule_desc = f"monthly on day 28 (fallback for 'last') at {hour:02d}:{minute:02d}"
            else:
                job.setall(f'{minute} {hour} {day_monthly} * *'); schedule_desc = f"monthly on day {day_monthly} at {hour:02d}:{minute:02d}"
        else:
            log_message(f"Unknown freq '{frequency}' for {log_prefix}. No job.", "ERROR"); return False
        is_valid_schedule = job.is_valid()
    except Exception as e_set_schedule:
         log_message(f"Error setting schedule ('{schedule_desc}') for {log_prefix}: {e_set_schedule}", "ERROR"); return False

    if is_valid_schedule:
        log_message(f"{log_prefix} cron job valid: '{command}' scheduled {schedule_desc}"); return True
    else:
        log_message(f"{log_prefix} cron job schedule invalid.", "ERROR")
        try: cron_instance.remove(job)
        except: pass # Already tried to log removal issues
        return False


def remove_rot_controller_schedule(cron_instance: CronTab):
    """Explicitly removes any old ROT controller cron jobs if they exist."""
    rot_cron_comment = "TenFinScraperJob_SiteController_ROT" # Use the old comment string
    log_message(f"Checking for and removing obsolete ROT controller cron jobs ('{rot_cron_comment}')...")
    try:
        jobs_to_remove = list(cron_instance.find_comment(rot_cron_comment))
        if jobs_to_remove:
            for job in jobs_to_remove:
                cron_instance.remove(job)
            log_message(f"Successfully removed {len(jobs_to_remove)} obsolete ROT controller cron jobs.")
        else:
            log_message("No obsolete ROT controller cron jobs found to remove.")
    except Exception as e_remove_rot:
        log_message(f"Error removing obsolete ROT controller cron jobs: {e_remove_rot}", "ERROR")


def set_cleanup_schedule(settings: dict, cron_instance: CronTab):
    retention_settings = settings.get("retention", {})
    is_enabled = retention_settings.get("enabled", False)
    log_message(f"Processing cleanup schedule: RetentionEnabled={is_enabled}")
    try:
        jobs_to_remove = list(cron_instance.find_comment(CLEANUP_CRON_COMMENT))
        if jobs_to_remove:
            for job in jobs_to_remove: cron_instance.remove(job)
            log_message(f"Removed {len(jobs_to_remove)} existing cleanup cron jobs: {CLEANUP_CRON_COMMENT}")
    except Exception as e_remove:
         log_message(f"Error removing cleanup cron jobs ({CLEANUP_CRON_COMMENT}): {e_remove}", "ERROR")
    if not is_enabled:
        log_message("Retention (cleanup) disabled. No cleanup cron job created.")
        return True
    dashboard_host_port = os.environ.get("TENFIN_DASHBOARD_HOST_PORT", "localhost:8081") 
    cleanup_endpoint = "/run-cleanup"; command = f"curl -sS -X POST http://{dashboard_host_port}{cleanup_endpoint} > /dev/null 2>&1"
    log_message(f"Constructed cleanup command: {command}")
    job: Optional[CronItem] = None
    try:
        job = cron_instance.new(command=command, comment=CLEANUP_CRON_COMMENT); job.setall('5 4 * * *')
        is_valid_schedule = job.is_valid()
        if is_valid_schedule: log_message(f"Cleanup cron job valid (Daily at 04:05)."); return True
        else: log_message("Cleanup cron job invalid.", "ERROR"); try: cron_instance.remove(job)
        except: pass; return False
    except Exception as e_new_job:
        log_message(f"Error creating/setting cleanup cron job: {e_new_job}", "ERROR"); return False


if __name__ == "__main__":
    log_message("Scheduler Setup Script Started")
    log_message("============================")
    # ... (user, cwd, python exec, script dir logging) ...

    settings_data = load_settings()
    if not settings_data:
        log_message("Exiting: settings load failure.", "CRITICAL"); sys.exit(1)

    cron: Optional[CronTab] = None
    try:
        current_user = getpass.getuser()
        cron = CronTab(user=current_user)
        log_message(f"Accessed crontab for '{current_user}'.")
    except Exception as e:
        log_message(f"ERROR: Could not access crontab for '{current_user}': {e}", "CRITICAL"); sys.exit(1)

    log_message("\nSetting REGULAR site_controller.py schedule...")
    success_controller_regular = set_regular_controller_schedule(settings_data, cron)
    log_message(f"Result of REGULAR controller schedule: {'Success' if success_controller_regular else 'Failure'}")

    # Remove any old ROT controller schedule as it's now manual
    remove_rot_controller_schedule(cron)
    log_message("ROT controller schedule is now manual; any existing ROT cron job has been removed.")
    success_controller_rot = True # Mark as success because removal is the intended action.

    log_message("\nSetting cleanup schedule...")
    success_cleanup = set_cleanup_schedule(settings_data, cron)
    log_message(f"Result of setting cleanup schedule: {'Success' if success_cleanup else 'Failure'}")

    try:
        log_message("\nAttempting to write changes to crontab...")
        # Always write if any main schedule change was attempted or cleanup change was attempted
        # or if ROT removal was performed.
        cron.write() 
        log_message("Crontab write command executed.")
        log_message("Verifying jobs written:")
        cron_verify = CronTab(user=current_user) 
        for job_item_verify in cron_verify: log_message(f"  - Verified Job: {job_item_verify}")
    except Exception as e_write:
        log_message(f"ERROR writing to crontab: {e_write}", "CRITICAL"); sys.exit(1)

    log_message("\n----------------------")
    if success_controller_regular and success_cleanup: # success_controller_rot is now always true if removal is successful
        log_message("Scheduler setup completed successfully based on settings (ROT is manual).")
        sys.exit(0)
    else:
        log_message("One or more scheduler tasks (Regular/Cleanup) may have failed. Review logs.", "ERROR")
        sys.exit(1)
