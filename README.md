# Climatrack

AI-powered climate impact calculator. Track your carbon footprint and get insights.

## Features

- **Calculator** – Describe your day; get carbon estimates (transport, electricity, food)
- **Analysis** – What-if optimizer (ZIP, commute, meals)
- **Insights** – Progress and carbon journey over time
- **News** – Climate news with AI opinions
- **Community** – Describe a problem → AI solution + Reddit posts
- **Risk** – Air quality, heat index, temperature (Open-Meteo)
- **Settings** – Location, units (Imperial/Metric)
- **Ask Climi** – Climate & environment Q&A + related articles

## Run locally

```bash
git clone https://github.com/SDRoan/ClimaTrack.git
cd ClimaTrack
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app_flask.py
```

Then open **http://127.0.0.1:5050** in your browser.
