
# News Scraper (SerpApi + newspaper3k)
This small project fetches Google News results via SerpApi, downloads article text (using newspaper3k or a BeautifulSoup fallback), runs a placeholder "AI" analysis (mocked), and writes results to an Excel file.
## Files
- `scrape_news.py` - main script
- `requirements.txt` - Python dependencies
- `.env.example` - example env file showing `SERPAPI_KEY`
## Setup (Windows PowerShell)
Open PowerShell and run the following commands in the project directory.
1) Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2) Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
3) Download NLTK punkt (required by newspaper3k)
```powershell
python -c "import nltk; nltk.download('punkt')"
```
4) Set your SerpApi key (replace with your actual key). You can set `SERPAPI_KEY` or `API_KEY` as environment variable.
(If you prefer, create a `.env` file and use a tool like `python-dotenv` to load it. The script currently reads from environment variables only.)
## Run
- Sample Text
```powershell
python scrape_news.py --query "data center Georgia" --max 250 --output "250_datacenter_news_final.xlsx"
```
The script will create an Excel file at the path you provide (or `output.xlsx` by default).
## Notes & Tips
- The code uses the `serpapi` package. SerpApi is a paid API — make sure you have credits and follow their usage policies.
- The script includes a mocked `mock_ai_analysis` function. Replace this with your AI call if needed.
- If you encounter SSL / network errors, confirm your machine can reach the API endpoints and that no proxy/firewall is blocking requests.
- If `newspaper3k` fails to parse some pages, the BeautifulSoup fallback will attempt to extract `<p>` text.
## Optional improvements
- Add argument to read `.env` file automatically (via `python-dotenv`).
- Add retries with backoff for network calls.
- Replace `mock_ai_analysis` with a real analysis pipeline.
<img width="1109" height="1209" alt="image" src="https://github.com/user-attachments/assets/2b50c75a-62d2-4231-8efb-28e9832ba7e6" />
