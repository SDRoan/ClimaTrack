# 🌍 Climatrack - AI Climate Impact Calculator

A powerful AI-powered climate impact calculator. Track your carbon footprint, get personalized insights, find real solutions from the community, and ask an environment-focused AI assistant. **Built with Flask** (and optional Streamlit).

## ✨ Features

- **🏠 Calculator**: Describe your day in natural language; get AI analysis and region-aware carbon estimates (transport, electricity, food).
- **📊 Analysis**: Region-aware "what-if" optimizer (ZIP, commute, meals, efficiency).
- **📈 Insights**: Progress tracking and carbon journey over time.
- **📰 News**: Climate news with multiple AI agent opinions and voting.
- **👥 Community**: Describe your problem → AI-generated solution + Reddit posts (strict + related), with deep analysis and fast parallel fetch.
- **⚠️ Risk**: Air quality, heat index, and temperature anomaly (Open-Meteo) with composite risk score.
- **⚙️ Settings**: Location, units (Imperial/Metric), notifications.
- **🤖 Ask Climi**: Climate & environment Q&A only; returns an answer plus related web articles.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- (Optional) [Ollama](https://ollama.ai) for local LLM (otherwise uses FLAN-T5)

### Installation

```bash
git clone https://github.com/SDRoan/ClimaTrack.git
cd ClimaTrack
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the app (Flask – recommended)

```bash
python app_flask.py
```

Open **http://127.0.0.1:5050**

### Run with Streamlit (optional)

```bash
streamlit run app.py
```

Open **http://localhost:8501**

## 🏗️ Structure

- **`app_flask.py`** – Flask app and API routes (calculator, news, community, risk, settings, Ask Climi).
- **`core.py`** – Shared logic: LLM (Ollama/FLAN-T5), carbon math, Reddit search, web search, risk/weather, news opinions.
- **`templates/index.html`** – Single-page UI (tabs, forms, Earth animation, custom icons).
- **`app.py`** – Legacy Streamlit UI (optional).
- **`data/`** – Optional CSVs for grid factors and footprint log (gitignored).

## 🔧 Configuration

- **Ollama** (optional): Set `OLLAMA_HOST` and `OLLAMA_MODEL` in env or use defaults (`http://localhost:11434`, `llama3.2:3b`).
- **Port**: `PORT=5050` for Flask (default 5050).
- **Data**: Add `data/egrid_subregion_factors.csv` and `data/zip_to_egrid.csv` for finer region-aware calculations.

## 📄 License

MIT. **Made with ❤️ for the planet.**
