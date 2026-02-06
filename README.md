# Climatrack

AI-powered climate impact calculator. Track your carbon footprint and get insights.

## Features

- **Calculator** – Describe your day; get carbon estimates (transport, electricity, food)
- **Analysis** – What-if optimizer (ZIP, commute, meals)
- **Insights** – Progress and carbon journey over time
- **News** – Climate news with AI opinions<img width="731" height="726" alt="Screenshot 2026-02-06 at 1 16 15 AM" src="https://github.com/user-attachments/assets/5b65d0ed-bbbb-4f4f-898e-489d1cd51dec" />

- **Community** – Describe a problem → AI solution + Reddit posts
- **Risk** – Air quality, heat index, temperature (Open-Meteo)
- **Settings** – Location, units (Imperial/Metric)
- **Ask Climi** – Climate & environment Q&A + related articles

## Run locally

```bash
git clone https://github.com/SDRoan/ClimaTrack.git
cd ClimaTrack
python -m venv .venv![Uploading image.png…]()

source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app_flask.py<img width="731" height="726" alt="Screenshot 2026-02-06 at 1 16 15 AM" src="https://github.com/user-attachments/assets/500033d2-6309-4dc7-bc63-94704a9b4a88" />

```

Then open **http://127.0.0.1:5050** in your browser.
