#!/bin/bash

# TenFin Setup Script
# ASSUMPTION: A Python 3 virtual environment is ALREADY CREATED AND ACTIVATED
#             before running this script.

# --- Configuration ---
SERVICE_NAME="tenfin"
SERVICE_DESCRIPTION="TenFin Tender Dashboard Service"
# !!! IMPORTANT: Set the user/group the service should run as !!!
#     Ensure this user has read/write permissions for logs/data dirs
#     and read/execute permissions for the project files/venv.
#     If project is in /root, running as non-root requires permission adjustments.
SERVICE_USER="root" # Or root, or another user
SERVICE_GROUP="root" # Or root, or another group
DASHBOARD_PORT="8081" # Match the port uvicorn will use

# --- Helper Functions ---
log_msg() {
    local type="$1"
    local msg="$2"
    echo "[$type] $msg"
}

check_command() {
    local cmd="$1"
    if ! command -v "$cmd" &> /dev/null; then
        log_msg "ERROR" "'$cmd' command not found. Please install it."
        exit 1
    fi
}

# --- Safety Checks ---
set -e # Exit immediately if a command exits with a non-zero status.

# --- Identify Project Directory ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$SCRIPT_DIR"
log_msg "INFO" "Project directory identified as: $PROJECT_DIR"

# --- Check for Root Privileges (Needed for systemd and apt) ---
log_msg "INFO" "Checking for root privileges..."
if [ "$(id -u)" -ne 0 ]; then
    log_msg "ERROR" "This script requires root privileges to install system dependencies and set up the systemd service. Please run with sudo."
    exit 1
else
    log_msg "SUCCESS" "Running as root."
fi

# --- Check Essential Commands ---
log_msg "INFO" "Checking for essential commands..."
check_command python3
check_command systemctl

# Determine pip command
if command -v python3 -m pip &> /dev/null; then
    PIP_CMD="python3 -m pip"
else
    check_command pip3 # Fallback, less preferred
    PIP_CMD="pip3"
fi
log_msg "SUCCESS" "'$PIP_CMD' will be used for package installation."
log_msg "SUCCESS" "'systemctl' found. Systemd setup will be attempted."


# --- Check System Dependencies (Example for Debian/Ubuntu/DietPi) ---
# Add more dependencies here if needed by playwright or other libs
REQUIRED_SYS_DEPS=(
    # Basic build tools often needed
    # build-essential
    # python3-dev
    # Add others as identified by errors or playwright install-deps
)
log_msg "INFO" "Checking system dependencies (APT package manager assumed)..."
INSTALL_NEEDED=()
for pkg in "${REQUIRED_SYS_DEPS[@]}"; do
    if ! dpkg -s "$pkg" &> /dev/null; then
        log_msg "WARNING" "System dependency '$pkg' seems missing."
        INSTALL_NEEDED+=("$pkg")
    fi
done

if [ ${#INSTALL_NEEDED[@]} -ne 0 ]; then
    log_msg "INFO" "Attempting to install missing system dependencies: ${INSTALL_NEEDED[*]}"
    if apt-get update && apt-get install -y "${INSTALL_NEEDED[@]}"; then
        log_msg "SUCCESS" "Missing system dependencies installed."
    else
        log_msg "ERROR" "Failed to install system dependencies. Please install manually: sudo apt-get install ${INSTALL_NEEDED[*]}"
        exit 1
    fi
else
     log_msg "SUCCESS" "Required system dependencies appear installed."
fi

# --- VIRTUAL ENVIRONMENT CHECKS REMOVED ---
# Script now assumes venv is active.

# --- Install Python Packages ---
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
log_msg "INFO" "Installing Python packages from '$REQUIREMENTS_FILE'..."
log_msg "WARNING" "This will install packages into the currently active Python environment."
log_msg "IMPORTANT" "Ensure you have activated the correct virtual environment BEFORE running this script."

if [ -f "$REQUIREMENTS_FILE" ]; then
    if $PIP_CMD install -r "$REQUIREMENTS_FILE"; then
        log_msg "SUCCESS" "Python packages installed."
    else
        log_msg "ERROR" "Failed to install Python packages from $REQUIREMENTS_FILE."
        exit 1
    fi
else
    log_msg "ERROR" "requirements.txt not found in $PROJECT_DIR."
    exit 1
fi

# --- Install Playwright Browsers ---
log_msg "INFO" "Installing Playwright browser binaries..."
# Run playwright install within the active environment
if python3 -m playwright install --with-deps; then
    # Note: We check the command's exit code, not Playwright's internal dependency check warnings
    log_msg "SUCCESS" "Playwright browser installation command executed (check output above for dependency warnings)."
else
    log_msg "ERROR" "Playwright browser installation command failed."
    log_msg "INFO" "Attempting 'python3 -m playwright install-deps' to fix system dependencies..."
    # Try installing system deps via playwright
    if python3 -m playwright install-deps; then
         log_msg "INFO" "Attempted Playwright dependency installation. Trying browser install again..."
         if python3 -m playwright install --with-deps; then
              log_msg "SUCCESS" "Playwright browser installation command succeeded after dependency install."
         else
              log_msg "ERROR" "Playwright browser installation failed even after attempting dependency install. Manual intervention needed."
              exit 1
         fi
    else
         log_msg "ERROR" "Playwright dependency installation failed. Manual intervention needed."
         exit 1
    fi
fi


# --- Setup Systemd Service ---
log_msg "INFO" "Setting up systemd service '$SERVICE_NAME'..."

# Find the uvicorn executable *within the active environment*
UVICORN_EXEC_PATH=$(command -v uvicorn)
if [ -z "$UVICORN_EXEC_PATH" ]; then
    log_msg "ERROR" "Could not find 'uvicorn' executable in the current environment's PATH."
    log_msg "ERROR" "Ensure your virtual environment is activated and uvicorn is installed correctly."
    exit 1
fi
log_msg "INFO" "Found uvicorn executable at: $UVICORN_EXEC_PATH"

# Create service file content
SERVICE_FILE_CONTENT="[Unit]
Description=$SERVICE_DESCRIPTION
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=$PROJECT_DIR
ExecStart=$UVICORN_EXEC_PATH dashboard:app --host 0.0.0.0 --port $DASHBOARD_PORT
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
"

SERVICE_FILE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

log_msg "INFO" "Generating service file content..."
log_msg "INFO" "Service will be configured to run as user: $SERVICE_USER, group: $SERVICE_GROUP"

# Permission Warning (Still relevant)
if [[ "$PROJECT_DIR" == /root* ]] && [[ "$SERVICE_USER" != "root" ]]; then
    log_msg "WARNING" "Project is in '$PROJECT_DIR' but service runs as '$SERVICE_USER'."
    log_msg "WARNING" "Ensure user '$SERVICE_USER' has necessary permissions for data/log directories within $PROJECT_DIR."
fi

log_msg "INFO" "Writing service file to $SERVICE_FILE_PATH..."
echo "$SERVICE_FILE_CONTENT" > "$SERVICE_FILE_PATH"

log_msg "INFO" "Setting permissions for $SERVICE_FILE_PATH..."
chmod 644 "$SERVICE_FILE_PATH"

log_msg "INFO" "Reloading systemd daemon..."
systemctl daemon-reload

log_msg "INFO" "Enabling service $SERVICE_NAME to start on boot..."
systemctl enable "$SERVICE_NAME"

log_msg "INFO" "Attempting to start service $SERVICE_NAME now..."
if systemctl start "$SERVICE_NAME"; then
    log_msg "SUCCESS" "Systemd service setup process finished."
else
    log_msg "ERROR" "Failed to start service $SERVICE_NAME. Check status with 'systemctl status $SERVICE_NAME' and logs with 'journalctl -u $SERVICE_NAME'."
    # Optionally exit here, or let the script finish
    # exit 1
fi


# --- Final Instructions ---
log_msg "SUCCESS" "Setup script finished!"
echo "--------------------------------------------------"
log_msg "IMPORTANT" "This script assumed a Python virtual environment was ALREADY ACTIVE."
log_msg "IMPORTANT" "Packages were installed into the active environment: $(command -v python3)"
echo ""
log_msg "INFO" "To run scrapers manually:"
log_msg "INFO" "1. Activate the correct Python environment (e.g., 'source venv/bin/activate')."
log_msg "INFO" "2. Navigate to the project directory: cd $PROJECT_DIR"
log_msg "INFO" "3. Run: python site_controller.py"
echo ""
log_msg "INFO" "If systemd setup was performed:"
log_msg "INFO" " - The dashboard service name is '$SERVICE_NAME'."
log_msg "INFO" " - Access it at: http://<YOUR_SERVER_IP>:$DASHBOARD_PORT"
log_msg "INFO" " - Check status: sudo systemctl status $SERVICE_NAME"
log_msg "INFO" " - Stop service: sudo systemctl stop $SERVICE_NAME"
log_msg "INFO" " - Start service: sudo systemctl start $SERVICE_NAME"
log_msg "INFO" " - View logs:    sudo journalctl -u $SERVICE_NAME -f"
echo "--------------------------------------------------"

exit 0
