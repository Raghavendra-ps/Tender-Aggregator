# TenFin - Tender Aggregation and Filtering System

TenFin is designed to scrape tender information from various Indian government eProcurement portals, aggregate the data, and provide a web-based dashboard for filtering, viewing, and managing these tenders.

## Features

*   **Multi-Site Scraping**: Capable of fetching data from numerous (configurable) NIC eProcurement portals.
*   **Automated Scheduling**: Schedules regular fetching runs for tender data and site configurations.
*   **Data Aggregation**: Merges tender data from all enabled sites into a single, consolidated list.
*   **Powerful Filtering Engine**:
    *   Filter tenders by keywords (simple string match or Regex).
    *   Filter by source site/state.
    *   Filter by tender opening date range.
*   **Web Dashboard (FastAPI & Jinja2)**:
    *   View lists of filtered tender sets.
    *   View detailed information for individual tenders.
    *   Download filtered tender sets in Excel (.xlsx) format (single or bulk).
    *   Manage (delete) filtered tender sets.
    *   Configure global settings (concurrency, timeouts, retries).
    *   Enable/disable specific tender portals.
    *   Configure and apply schedules.
    *   Manage data retention policies for old filtered results.
    *   View logs (site controller and individual site logs).
    *   Manually trigger fetch runs.

*   **Structured Data Extraction**: Parses HTML to extract detailed tender information, including critical dates, fees, work details, and associated documents.
*   **Modular Design**: Separates concerns into scraping logic, orchestration, filtering, and the web dashboard.

## Technical Stack

*   **Backend**: Python
*   **Web Framework**: FastAPI
*   **Templating**: Jinja2
*   **Web Scraping**: Playwright, BeautifulSoup4
*   **Scheduling**: `python-crontab` (interfacing with system `cron`)
*   **Data Handling**: JSON, OpenPyXL (for Excel export)

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd TenFin-main
    ```

2.  **Create a Virtual Environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Playwright Browsers**:
    Playwright requires browser binaries to be installed.
    ```bash
    python -m playwright install --with-deps
    # Or, to install only chromium:
    # python -m playwright install chromium
    ```

5.  **Configure `settings.json`**:
    *   Review and update `TenFin-main/settings.json`.
    *   **`global_scraper_settings`**: Adjust concurrency, timeouts, and retry limits as needed for your network and system resources.
    *   **`retention`**: Configure if and how long filtered results should be kept.
    *   **`scheduler`**: Define the default schedule for the main scraper.
    *   **`site_configurations`**: This is crucial. Add or modify entries for the eProcurement portals you want to scrape.
        *   `"base_url"`: The URL pattern for fetching tender list pages (usually with `{}` for the page number).
        *   `"domain"`: The base domain of the portal, used for resolving relative links.
        *   `"enabled"`: Set to `true` to scrape this site, `false` to disable it.
        Example for a site:
        ```json
        "MyStateTenders": {
          "base_url": "https://tenders.mystate.gov.in/nicgep/app?component=%24TablePages.linkPage&page=FrontEndAdvancedSearchResult&service=direct&session=T&sp=AFrontEndAdvancedSearchResult%2Ctable&sp={}",
          "domain": "https://tenders.mystate.gov.in/nicgep/app?",
          "enabled": true
        }
        ```

6.  **Ensure `cron` Service is Running**:
    This application uses the system `cron` daemon for scheduling. Make sure it's installed and running on your server (typically Linux/macOS).

7.  **Start setup.sh**:
    This application uses the system `cron` daemon for scheduling. Make sure it's installed and running on your server (typically Linux/macOS).
    
## Running the Application

1.  **Start the Web Dashboard**:
    The dashboard provides the UI for configuration, filtering, and viewing. The commands used to operate this are :
    ```sudo systemctl start tenfin.service
    ```
     ```sudo systemctl restart tenfin.service
    ```
      ```sudo systemctl stop tenfin.service
    ```
      ```sudo systemctl status tenfin.service
    ```
       ```sudo systemctl disable tenfin.service
    ```
        ```sudo systemctl enable tenfin.service
    ```

    Access the dashboard in your browser, typically at `http://localhost:8081` or `http://<your-server-ip>:8081`.

3.  **Configure via Dashboard**:
    *   Navigate to the "Settings" page in the dashboard.
    *   **Enable Sites**: Select which tender portals to scrape.
    *   **Scraper Parameters**: Adjust performance settings if necessary.
    *   **Scheduling**:
        *   Set the desired frequency and time for the main scraper.
        *   Enable/disable data retention.
        *   Click "Save Schedule Settings" for each section you modify.
        *   Finally, click **"Apply Schedule to System (Cron)"**. This will run `scheduler_setup.py` to update your system's cron jobs based on all saved settings.

4.  **Manual Scraper Trigger**:
    *   You can trigger an immediate scraping run for all enabled sites by clicking the "Fetch Tenders Now" button on the dashboard's homepage.
    *   Alternatively, run `site_controller.py` directly (ensure your virtual environment is active):
        ```bash
        python site_controller.py
        ```

## Using the Filter Engine

1.  Once the scraper has run (either manually or via schedule) and `Final_Tender_List_*.txt` exists in `scraped_data/`, go to the dashboard.
2.  Click on "+ New Filter".
3.  Provide:
    *   **Filter Name**: A descriptive name for your saved filter results.
    *   **Source Site (Optional)**: Select a specific state/portal or leave as "Any Site".
    *   **Opening Date Range (Optional)**: Specify a date range for tender opening.
    *   **Keywords (Optional)**: Comma-separated terms to search for.
    *   **Use Regex**: Check if your keywords are regular expressions.
4.  Click "Run Filter". Results will be saved in `scraped_data/Filtered Tenders/<Filter_Name>_Tenders/`.
5.  You can then view or download these results from the dashboard homepage.

## Logging

*   **Main Orchestrator**: `logs/site_controller.log` shows the high-level progress of `site_controller.py`.
*   **Individual Sites**: `logs/scrape_<SiteKey>.log` contains detailed logs for the scraping process of each specific site.
*   Logs can be viewed and downloaded from the "View Logs" section on the dashboard's "Settings" page.

## Troubleshooting

*   **Playwright Issues**: If you encounter errors related to Playwright (e.g., "browser not found"), ensure you've run `python -m playwright install --with-deps` correctly.
*   **Cron Job Not Running**:
    *   Verify `cron` service is active on your system.
    *   Check system logs (e.g., `/var/log/syslog` or `/var/log/cron`) for errors related to the cron jobs set up by `scheduler_setup.py`.
    *   Ensure the Python executable path (`sys.executable` used in `scheduler_setup.py`) is correct for the cron environment and that the script has execution permissions.
    *   The cron job runs `cd /path/to/TenFin-main && /path/to/venv/bin/python site_controller.py`. Make sure these paths are correct in the crontab entry (`crontab -l`).
*   **Settings Not Taking Effect**: After changing settings in `settings.json` or the dashboard, remember to:
    *   Save the specific setting section in the dashboard.
    *   Click "Apply Schedule to System (Cron)" if you changed scheduling parameters, enabled/disabled sites that affect scheduling, or retention settings (as cleanup is also scheduled).
*   **No Data Scraped**:
    *   Check `logs/site_controller.log` and relevant `logs/scrape_<SiteKey>.log` for errors.
    *   Ensure the target sites are enabled in settings.
    *   Verify the `base_url` and `domain` in `settings.json` for the sites are still valid (websites change).
    *   The scraper might need adjustments if the target website's HTML structure has changed significantly.

# Tender-Aggregator
