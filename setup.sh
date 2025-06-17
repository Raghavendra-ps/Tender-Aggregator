#!/bin/bash

# TenFin Setup Script
#
# IMPORTANT ASSUMPTIONS:
# 1. This script is run WITH SUDO or AS ROOT due to system package installation
#    and systemd service management.
# 2. If you intend to use a Python virtual environment (RECOMMENDED):
#    - CREATE AND ACTIVATE the virtual environment *BEFORE* running this script.
#    - The script will then install Python packages into that active venv.
#    - The systemd service will be configured to use the Python from this venv.
# 3. If NOT using a virtual environment (NOT RECOMMENDED for system-wide installs):
#    - Python packages will be installed for the system Python3 used by root.
#
# Fedora 40/41/42 and Debian/Ubuntu (Recent LTS) Compatibility Focus

# --- Configuration ---
SERVICE_NAME="tenfin"
SERVICE_DESCRIPTION="TenFin Tender Dashboard and Scraper Service"

# !!! IMPORTANT: User/Group for the systemd service !!!
# This user needs:
#   - Read/execute access to the project directory and Python executable (venv or system).
#   - Read/write access to data directories (site_data/*) and log directories (LOGS/*).
#   - If this user is NOT root, ensure permissions are set accordingly *after* project setup.
#   - For cron jobs (scheduler_setup.py), this user's crontab will be modified.
SERVICE_USER="root"   # E.g., "tenfin_user" or "root"
SERVICE_GROUP="root"  # E.g., "tenfin_group" or "root"

DASHBOARD_PORT="8082" # Port for the FastAPI dashboard

# --- Helper Functions ---
log_msg() {
    local type="$1"
    local msg="$2"
    # Simple color for INFO, SUCCESS, WARNING, ERROR
    case "$type" in
        INFO)    echo -e "\033[0;34m[INFO]\033[0m $msg" ;;
        SUCCESS) echo -e "\033[0;32m[SUCCESS]\033[0m $msg" ;;
        WARNING) echo -e "\033[0;33m[WARNING]\033[0m $msg" ;;
        ERROR)   echo -e "\033[0;31m[ERROR]\033[0m $msg" ;;
        *)       echo "[$type] $msg" ;;
    esac
}

check_command_fatal() {
    local cmd="$1"
    local purpose="$2"
    if ! command -v "$cmd" &> /dev/null; then
        log_msg "ERROR" "'$cmd' command not found. It is required for $purpose. Please install it."
        exit 1
    fi
}

# --- Safety Checks ---
set -e # Exit immediately if a command exits with a non-zero status.
set -o pipefail # Causes a pipeline to return the exit status of the last command in the pipe that returned a non-zero return value.

# --- Identify Project Directory and Python Executable ---
SCRIPT_DIR_REAL="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$SCRIPT_DIR_REAL"
log_msg "INFO" "Project directory identified as: $PROJECT_DIR"

# Determine Python executable (respects active virtual environment)
PYTHON_EXEC=""
PIP_EXEC=""
if [[ -n "$VIRTUAL_ENV" ]]; then
    PYTHON_EXEC="$VIRTUAL_ENV/bin/python3"
    PIP_EXEC="$VIRTUAL_ENV/bin/pip3" 
    log_msg "INFO" "Active Python virtual environment detected: $VIRTUAL_ENV"
    log_msg "INFO" "Using Python from venv: $PYTHON_EXEC"
    if [ ! -f "$PIP_EXEC" ]; then # Check if pip3 executable exists in venv
        log_msg "WARNING" "Direct pip3 executable '$PIP_EXEC' not found in venv. Will rely on 'python -m pip'."
        PIP_EXEC="" # Unset it so the logic below tries python -m pip
    fi
else
    log_msg "WARNING" "No active Python virtual environment detected."
    log_msg "WARNING" "Python packages will be installed for the system Python 3."
    log_msg "WARNING" "It is STRONGLY recommended to use a virtual environment."
    if command -v python3 &> /dev/null; then
        PYTHON_EXEC="$(command -v python3)"
    else
        log_msg "ERROR" "python3 command not found. Please install Python 3."
        exit 1
    fi
    if command -v pip3 &> /dev/null; then
        PIP_EXEC="$(command -v pip3)"
    else
        log_msg "WARNING" "pip3 command not found. Attempting to install python3-pip..."
        set +e # Temporarily allow failure
        if command -v dnf &> /dev/null; then
            dnf install -y python3-pip &> /dev/null
        elif command -v apt-get &> /dev/null; then
            apt-get update -y -qq &> /dev/null
            apt-get install -y python3-pip &> /dev/null
        fi
        set -e # Re-enable exit on error
        if command -v pip3 &> /dev/null; then
            PIP_EXEC="$(command -v pip3)"
            log_msg "SUCCESS" "python3-pip installed for system Python."
        else
            log_msg "ERROR" "pip3 command still not found for system Python. Please ensure pip is installed."
            # PIP_EXEC will remain empty
        fi
    fi
    log_msg "INFO" "Using system Python: $PYTHON_EXEC"
fi

if [ ! -x "$PYTHON_EXEC" ]; then
    log_msg "ERROR" "Python executable '$PYTHON_EXEC' is not executable or not found."
    exit 1
fi

# Declare PIP_CMD_ARRAY as an array
declare -a PIP_CMD_ARRAY

# Determine the correct PIP command
log_msg "INFO" "Determining pip command..."
if [ -n "$PIP_EXEC" ] && [ -x "$PIP_EXEC" ] && "$PIP_EXEC" --version &> /dev/null; then
    log_msg "INFO" "Using direct pip executable: $PIP_EXEC"
    PIP_CMD_ARRAY=("$PIP_EXEC") 
elif "$PYTHON_EXEC" -m pip --version &> /dev/null; then
    log_msg "INFO" "Using Python module pip: $PYTHON_EXEC -m pip"
    PIP_CMD_ARRAY=("$PYTHON_EXEC" "-m" "pip") 
else
    log_msg "ERROR" "Could not find a working pip command. Neither '$PIP_EXEC' (if found) nor '$PYTHON_EXEC -m pip' are functional."
    log_msg "ERROR" "Please ensure pip is correctly installed for the Python environment at '$PYTHON_EXEC'."
    if [[ -n "$VIRTUAL_ENV" ]]; then
        log_msg "ERROR" "Try reactivating the virtual environment. If pip is missing, try: '$PYTHON_EXEC -m ensurepip --upgrade' or recreate the venv."
    fi
    exit 1
fi
log_msg "INFO" "Pip command to be used: ${PIP_CMD_ARRAY[*]}"


# --- Check for Root Privileges ---
log_msg "INFO" "Checking for root privileges..."
if [ "$(id -u)" -ne 0 ]; then
    log_msg "ERROR" "This script requires root privileges for system tasks. Please run with sudo."
    exit 1
fi
log_msg "SUCCESS" "Running with root privileges."

# --- Check Essential System Commands ---
log_msg "INFO" "Checking for essential system commands..."
check_command_fatal systemctl "managing systemd services"
check_command_fatal id "checking user ID"
check_command_fatal groupadd "creating service group (if needed)"
check_command_fatal useradd "creating service user (if needed)"
check_command_fatal mkdir "creating directories"
check_command_fatal chmod "setting permissions"
check_command_fatal chown "setting ownership"
check_command_fatal crontab "managing cron jobs for scheduler"

# --- Determine Package Manager & OS ---
log_msg "INFO" "Detecting operating system and package manager..."
OS_ID=""
if [ -f /etc/os-release ]; then
    # shellcheck source=/dev/null
    . /etc/os-release
    OS_ID=$ID
fi

PKG_MANAGER=""
INSTALL_CMD=""
UPDATE_CMD=""
CHECK_PKG_INSTALLED_CMD=""

case "$OS_ID" in
    fedora)
        PKG_MANAGER="dnf"
        INSTALL_CMD="dnf install -y --setopt=install_weak_deps=False"
        UPDATE_CMD="dnf check-update -y || true"
        CHECK_PKG_INSTALLED_CMD="rpm -q"
        log_msg "SUCCESS" "OS detected: Fedora. Package manager: $PKG_MANAGER"
        ;;
    ubuntu|debian|raspbian)
        PKG_MANAGER="apt-get"
        INSTALL_CMD="apt-get install -y --no-install-recommends"
        UPDATE_CMD="apt-get update -y -qq"
        CHECK_PKG_INSTALLED_CMD="dpkg-query -W -f='\${Status}'"
        log_msg "SUCCESS" "OS detected: $OS_ID. Package manager: $PKG_MANAGER"
        ;;
    *)
        log_msg "ERROR" "Unsupported OS: '$OS_ID'. This script supports Fedora and Debian-based systems (Ubuntu, etc.)."
        exit 1
        ;;
esac
check_command_fatal "$PKG_MANAGER" "installing system packages"


# --- System Dependencies for Python Package Building & Playwright ---
log_msg "INFO" "Defining system dependencies..."
FEDORA_DEPS=(
    python3-devel gcc gcc-c++ libffi-devel openssl-devel redhat-rpm-config
    libicu libjpeg-turbo libwebp harfbuzz enchant2
    alsa-lib at-spi2-atk cups-libs gtk3 libX11 libXcomposite libXdamage
    libXext libXfixes libXrandr libXScrnSaver libXtst mesa-libgbm nss pango pipewire-libs xdg-utils
)
DEBIAN_DEPS=(
    python3-dev build-essential libffi-dev libssl-dev pkg-config
    libicu-dev libjpeg-turbo8-dev libwebp-dev libharfbuzz-dev libenchant-2-dev
    libasound2 libatk-bridge2.0-0 libcups2 libgtk-3-0 libx11-6 libxcomposite1 libxdamage1
    libxext6 libxfixes3 libxrandr2 libxss1 libxtst6 libgbm1 libnss3 libpango-1.0-0 xdg-utils
)

REQUIRED_SYS_DEPS=()
if [ "$PKG_MANAGER" = "dnf" ]; then
    REQUIRED_SYS_DEPS=("${FEDORA_DEPS[@]}")
else
    REQUIRED_SYS_DEPS=("${DEBIAN_DEPS[@]}")
fi

log_msg "INFO" "Checking and installing system dependencies using $PKG_MANAGER..."
INSTALL_NEEDED=()
for pkg_spec in "${REQUIRED_SYS_DEPS[@]}"; do
    pkg_to_check="$pkg_spec"
    is_installed=false
    if [ "$PKG_MANAGER" = "dnf" ]; then
        if $CHECK_PKG_INSTALLED_CMD "$pkg_to_check" &> /dev/null; then
            is_installed=true
        fi
    else # apt-get
        if $CHECK_PKG_INSTALLED_CMD "$pkg_to_check" 2>/dev/null | grep -q "ok installed" &> /dev/null; then
            is_installed=true
        fi
    fi

    if ! $is_installed; then
        log_msg "INFO" "System dependency '$pkg_spec' will be installed."
        INSTALL_NEEDED+=("$pkg_spec")
    else
        log_msg "INFO" "System dependency '$pkg_to_check' appears to be installed."
    fi
done

if [ ${#INSTALL_NEEDED[@]} -ne 0 ]; then
    log_msg "INFO" "Updating package lists (may take a moment)..."
    if ! $UPDATE_CMD; then
        log_msg "WARNING" "Package list update failed, attempting dependency installation anyway."
    fi
    log_msg "INFO" "Installing missing system dependencies: ${INSTALL_NEEDED[*]}"
    if $INSTALL_CMD "${INSTALL_NEEDED[@]}"; then
        log_msg "SUCCESS" "System dependencies installed."
    else
        log_msg "ERROR" "Failed to install some system dependencies: ${INSTALL_NEEDED[*]}"
        log_msg "ERROR" "Please try installing them manually and re-run the script."
        exit 1
    fi
else
    log_msg "SUCCESS" "All specified system dependencies appear to be installed."
fi


# --- Install/Upgrade Pip and Install Python Packages ---
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
log_msg "INFO" "Ensuring pip is up-to-date and installing Python packages from '$REQUIREMENTS_FILE'..."
# PIP_CMD_ARRAY was determined earlier

if ! "${PIP_CMD_ARRAY[@]}" install --upgrade pip; then
    log_msg "WARNING" "Failed to upgrade pip. Continuing with existing version. This might cause issues."
fi

if [ -f "$REQUIREMENTS_FILE" ]; then
    if "${PIP_CMD_ARRAY[@]}" install -r "$REQUIREMENTS_FILE"; then
        log_msg "SUCCESS" "Python packages installed successfully."
    else
        log_msg "ERROR" "Failed to install Python packages from '$REQUIREMENTS_FILE'."
        exit 1
    fi
else
    log_msg "ERROR" "'$REQUIREMENTS_FILE' not found."
    exit 1
fi

# --- Install Playwright Browsers ---
log_msg "INFO" "Installing Playwright browser binaries (Chromium)..."
log_msg "INFO" "System dependencies for Playwright should have been installed in the previous step."
if "$PYTHON_EXEC" -m playwright install chromium; then
    log_msg "SUCCESS" "Playwright browser (Chromium) binaries installed/verified."
else
    log_msg "ERROR" "Playwright browser binary installation failed."
    log_msg "ERROR" "Try running '$PYTHON_EXEC -m playwright install chromium' manually."
    log_msg "ERROR" "Ensure system dependencies from the OS_DEPS list in this script were installed correctly by $PKG_MANAGER."
    log_msg "ERROR" "If issues persist, check Playwright's manual installation guide for your OS."
    exit 1
fi

# --- Create Service User and Group (if they don't exist and are not root) ---
if [ "$SERVICE_USER" != "root" ]; then
    log_msg "INFO" "Setting up service user '$SERVICE_USER' and group '$SERVICE_GROUP'..."
    if ! getent group "$SERVICE_GROUP" >/dev/null; then
        if groupadd -r "$SERVICE_GROUP"; then
            log_msg "SUCCESS" "Group '$SERVICE_GROUP' created."
        else
            log_msg "ERROR" "Failed to create group '$SERVICE_GROUP'."
            exit 1
        fi
    else
        log_msg "INFO" "Group '$SERVICE_GROUP' already exists."
    fi

    if ! id "$SERVICE_USER" >/dev/null 2>&1; then
        if useradd -r -s /usr/sbin/nologin -g "$SERVICE_GROUP" -d "$PROJECT_DIR" "$SERVICE_USER"; then
            log_msg "SUCCESS" "User '$SERVICE_USER' created."
        else
            log_msg "ERROR" "Failed to create user '$SERVICE_USER'."
            exit 1
        fi
    else
        log_msg "INFO" "User '$SERVICE_USER' already exists."
        if ! groups "$SERVICE_USER" | grep -q "\b$SERVICE_GROUP\b"; then
            log_msg "INFO" "Adding user '$SERVICE_USER' to group '$SERVICE_GROUP'."
            if ! usermod -a -G "$SERVICE_GROUP" "$SERVICE_USER"; then
                log_msg "WARNING" "Failed to add user '$SERVICE_USER' to group '$SERVICE_GROUP'. Permissions might need manual adjustment."
            fi
        fi
    fi
else
    log_msg "INFO" "Service will run as root. Skipping dedicated user/group creation."
fi


# --- Create Directories and Set Permissions ---
LOG_DIR="$PROJECT_DIR/LOGS"
SITE_DATA_DIR="$PROJECT_DIR/site_data"

log_msg "INFO" "Creating application directories..."
mkdir -p "$LOG_DIR/regular_scraper"
mkdir -p "$LOG_DIR/rot_worker" 

mkdir -p "$SITE_DATA_DIR/REG/RawPages"
mkdir -p "$SITE_DATA_DIR/REG/MergedSiteSpecific"
mkdir -p "$SITE_DATA_DIR/REG/FinalGlobalMerged"
mkdir -p "$SITE_DATA_DIR/REG/FilteredResults"
mkdir -p "$SITE_DATA_DIR/ROT/DetailHtmls" 
mkdir -p "$SITE_DATA_DIR/ROT/MergedSiteSpecific"
mkdir -p "$SITE_DATA_DIR/ROT/FinalGlobalMerged"
mkdir -p "$SITE_DATA_DIR/ROT/FilteredResults"
mkdir -p "$SITE_DATA_DIR/ROT/AI_Analysis"
mkdir -p "$SITE_DATA_DIR/TEMP/WorkerRuns"
mkdir -p "$SITE_DATA_DIR/TEMP/CaptchaImages"
mkdir -p "$SITE_DATA_DIR/TEMP/DebugScreenshots/dashboard"
mkdir -p "$SITE_DATA_DIR/TEMP/DebugScreenshots/worker" 


log_msg "INFO" "Setting permissions for application directories..."
if [ "$SERVICE_USER" != "root" ]; then
    chown -R "$SERVICE_USER":"$SERVICE_GROUP" "$PROJECT_DIR"
    find "$PROJECT_DIR" -type d -exec chmod 750 {} \; 
    find "$PROJECT_DIR" -type f -exec chmod 640 {} \; 
    find "$PROJECT_DIR" \( -name "*.py" -o -name "*.sh" \) -exec chmod 750 {} \;
    
    chown -R "$SERVICE_USER":"$SERVICE_GROUP" "$LOG_DIR" "$SITE_DATA_DIR"
    find "$LOG_DIR" -type d -exec chmod u=rwx,g=rx,o= {} \;
    find "$LOG_DIR" -type f -exec chmod u=rw,g=r,o= {} \;
    find "$SITE_DATA_DIR" -type d -exec chmod u=rwx,g=rwx,o= {} \; 
    find "$SITE_DATA_DIR" -type f -exec chmod u=rw,g=rw,o= {} \;   
else
    chmod -R u+rwX "$LOG_DIR"
    chmod -R u+rwX "$SITE_DATA_DIR"
fi
log_msg "SUCCESS" "Application directories created and permissions set."


# --- Setup Systemd Service ---
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
SERVICE_PYTHON_EXEC="$PYTHON_EXEC" 
UVICORN_EXEC_IN_VENV_PATH="${SERVICE_PYTHON_EXEC%/*}/uvicorn"

if [ ! -x "$UVICORN_EXEC_IN_VENV_PATH" ]; then
    log_msg "WARNING" "Uvicorn not found at '$UVICORN_EXEC_IN_VENV_PATH'."
    log_msg "WARNING" "Attempting to find uvicorn in PATH (this might use a system-wide uvicorn)."
    if ! command -v uvicorn &> /dev/null; then
        log_msg "ERROR" "uvicorn command not found. Please ensure it's installed in the Python environment: ${PIP_CMD_ARRAY[*]} install uvicorn"
        exit 1
    fi
    UVICORN_EXEC_IN_VENV_PATH=$(command -v uvicorn)
fi
log_msg "INFO" "Using uvicorn executable at: $UVICORN_EXEC_IN_VENV_PATH"


log_msg "INFO" "Creating systemd service file at $SERVICE_FILE..."
EXEC_START_COMMAND_STR="$SERVICE_PYTHON_EXEC -m uvicorn dashboard:app --host 0.0.0.0 --port $DASHBOARD_PORT --root-path /"
ENV_FILE_PATH="$PROJECT_DIR/.env"

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=$SERVICE_DESCRIPTION
After=network.target

[Service]
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$PROJECT_DIR

ExecStart=$EXEC_START_COMMAND_STR

Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# If your .env file exists and contains simple KEY=VALUE pairs, it might be picked up by EnvironmentFile
# The '-' prefix makes systemd ignore if the file is missing.
$( [ -f "$ENV_FILE_PATH" ] && echo "EnvironmentFile=-$ENV_FILE_PATH" )

# Explicitly set variables needed by the application if not in .env or for overrides
Environment="TENFIN_DASHBOARD_HOST_PORT=localhost:$DASHBOARD_PORT"
# Add other necessary environment variables here, e.g.:
# Environment="OPENWEBUI_API_BASE_URL=http://localhost:8080"
# Environment="OPENWEBUI_API_KEY_CONFIG=your_key_here" # IMPORTANT: Set your actual key!

[Install]
WantedBy=multi-user.target
EOF

chmod 644 "$SERVICE_FILE"

log_msg "SUCCESS" "Systemd service file created."
log_msg "INFO" "Reloading systemd daemon, enabling and starting service..."
if ! systemctl daemon-reload; then
    log_msg "ERROR" "Failed to reload systemd daemon."
    exit 1
fi
if ! systemctl enable "$SERVICE_NAME"; then
    log_msg "ERROR" "Failed to enable $SERVICE_NAME service."
    exit 1
fi

sleep 1 

if systemctl start "$SERVICE_NAME"; then
    log_msg "SUCCESS" "$SERVICE_NAME service started and enabled."
    log_msg "INFO" "You can check service status with: systemctl status $SERVICE_NAME"
    log_msg "INFO" "Logs can be viewed with: journalctl -u $SERVICE_NAME -f"
else
    log_msg "ERROR" "Failed to start $SERVICE_NAME service."
    log_msg "ERROR" "Please check service status and logs for more details:"
    log_msg "ERROR" "  sudo systemctl status $SERVICE_NAME"
    log_msg "ERROR" "  sudo journalctl -u $SERVICE_NAME -n 100 --no-pager"
    exit 1
fi

# --- Setup Cron Job for Scheduler ---
log_msg "INFO" "Setting up cron jobs via scheduler_setup.py..."
log_msg "INFO" "This will attempt to modify the crontab of user: $SERVICE_USER"
SCHEDULER_SETUP_SCRIPT="$PROJECT_DIR/scheduler_setup.py"

if [ -f "$SCHEDULER_SETUP_SCRIPT" ]; then
    CRON_SETUP_CMD_PREFIX=""
    if [ "$SERVICE_USER" != "root" ] && [ "$(id -u)" -eq 0 ]; then
        CRON_SETUP_CMD_PREFIX="sudo -u $SERVICE_USER "
        log_msg "INFO" "Will run scheduler_setup.py as user '$SERVICE_USER'."
    elif [ "$SERVICE_USER" = "root" ] && [ "$(id -u)" -ne 0 ]; then
        log_msg "ERROR" "Script running as non-root, but SERVICE_USER is root. Cron setup for root requires root privileges for this step."
    else
        log_msg "INFO" "Will run scheduler_setup.py as current user ($(whoami))."
    fi

    chmod u+x "$SCHEDULER_SETUP_SCRIPT" # Ensure script is executable by its owner

    # Execute scheduler_setup.py from the project directory
    # The SERVICE_PYTHON_EXEC should be the python from the venv
    # Use eval to correctly handle the command string with potential sudo prefix
    if eval "cd '$PROJECT_DIR' && ${CRON_SETUP_CMD_PREFIX} '$SERVICE_PYTHON_EXEC' '$SCHEDULER_SETUP_SCRIPT'"; then
        log_msg "SUCCESS" "Scheduler setup script executed. Check its output for cron job status details."
    else
        log_msg "ERROR" "Scheduler setup script execution failed. Exit code: $?"
        log_msg "ERROR" "Cron jobs for automated scraping might not be configured correctly."
        log_msg "ERROR" "Try running manually: cd '$PROJECT_DIR' && ${CRON_SETUP_CMD_PREFIX} '$SERVICE_PYTHON_EXEC' '$SCHEDULER_SETUP_SCRIPT'"
    fi
else
    log_msg "WARNING" "'$SCHEDULER_SETUP_SCRIPT' not found. Skipping cron job setup."
fi


log_msg "INFO" "-----------------------------------------------------------"
log_msg "INFO" "TenFin Setup Script Completed."
log_msg "INFO" "Dashboard should be accessible at: http://<your_server_ip>:$DASHBOARD_PORT/"
if [[ -n "$VIRTUAL_ENV" ]]; then
    log_msg "INFO" "Service is configured to use Python from venv: $SERVICE_PYTHON_EXEC"
fi
log_msg "INFO" "Data will be stored in: $SITE_DATA_DIR"
log_msg "INFO" "Service logs via 'journalctl -u $SERVICE_NAME'. Scraper-specific logs in: $LOG_DIR"
log_msg "INFO" "-----------------------------------------------------------"

exit 0
