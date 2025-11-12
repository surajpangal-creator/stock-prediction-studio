## Stock Prediction Web App (New)

The repository now includes a Flask-powered website for forecasting stock prices or volatility with data sourced from the Yahoo Finance API.

### Quick Start

```bash
python3 -m pip install -r requirements.txt
python3 stock_prediction_app/app.py
```

Then open your browser to `http://127.0.0.1:5000/` to access the dashboard.

### Features

- Choose a ticker, sampling interval, target (close price or rolling volatility), and forecast horizon.
- Run forecasts with either:
  - **Prophet** for probabilistic decomposable trend modeling.
  - **LSTM** neural networks for deep learning-based predictions.
- Review dark-mode summary dashboards featuring forecast tables, deltas, and evaluation metrics.
- Download-free experience powered by `yfinance` data.

### Deploy to Render (Demo Hosting)

1. Push this repository to a Git host (GitHub, GitLab, etc.).
2. The included `render.yaml` configures a Free-tier Python web service.
3. On [Render](https://render.com), create a **New Web Service** connected to your repo and use the defaults from `render.yaml`:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn stock_prediction_app.app:app`
4. Deploy—Render will build the container and provide a public URL you can share for demos.

Prophet forecasting requires `cmdstanpy` the first time it runs; the dependency is installed automatically via `pip`, but the first model fit may take a minute while the Stan backend is compiled.

---

## How to Run

There are three ways to use this application, depending on your preference.

### Option 1: Standalone Program (Recommended for Windows)

For most users, the easiest method is to use the pre-built `.exe` file. No Python installation is needed.

1.  Go to the [Releases page](https://github.com/yzRobo/draftkings_api_explorer/releases) on GitHub.
2.  Download the `DK_API_Scraper_vX.X.X.exe` file from the latest release.
3.  Run the downloaded file.

---

### Option 2: Running from Source (with `run.bat` for Windows)

This option is for users who download the source code and want a simple way to run it on Windows without using the command line.

1.  Make sure you have Python installed on your system.
2.  Download or clone the project repository.
3.  Install the required dependencies by opening a terminal or command prompt in the project folder and running:
    ```bash
    pip install -r requirements.txt
    ```
4.  Once dependencies are installed, simply double-click the `run.bat` file to start the application.

---

### Option 3: Running from Source (Manual)

This method works for all operating systems (Windows, macOS, Linux) and is the standard way to run a Python application from source.

1.  Ensure you have Python installed.
2.  Download or clone the project repository.
3.  It is highly recommended to create and activate a virtual environment:
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5.  Run the application from your terminal:
    ```bash
    python dk_api_gui_explorer.py
    ```

---

## Features

- **Graphical User Interface (GUI)** using Tkinter
- Supports multiple market categories including:
  - Wins
  - Futures
  - Awards
  - Player Futures
  - Rookie Watch
  - Stat Leaders
  - Playoffs
  - Division Specials
  - Team Specials
  - Game-Specific Props
- **Data parsing and pivoting** for Over/Under markets
- **Export to CSV**
- **Built-in ID reference** for league, category, and subcategory IDs
- Designed to work with the **DraftKings NFL League ID** (`88808`)

---

## Dependencies

- `tkinter` (included with most Python installations)
- `pandas`
- `curl_cffi`

---

---

## File Structure

```
draftkings_api_explorer/
│
├── .gitignore               # Specifies intentionally untracked files to ignore
├── config.json              # Configuration file for API endpoints
├── dk_api_gui_explorer.py   # The main application script with the GUI
├── id_reference.json        # Reference data for market categories and IDs
├── LICENSE                  # The MIT License for the project
├── README.md                # The project's documentation file
└── requirements.txt         # A list of the Python packages required
```

---

## Coming Soon / To Do

- Correct Formatting of more API categories to ensure output looks correct
- Add support for retrieving data across multiple subcategory IDs

---

## License

This project is open source under the MIT License.

---

## Author

yzRobo
