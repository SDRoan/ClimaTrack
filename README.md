# 🌍 Climatrack - AI Climate Impact Calculator

A powerful AI-powered climate impact calculator built with Streamlit. Track your carbon footprint, get personalized insights, and find real solutions from the community.

## ✨ Features

- **🤖 AI-Powered Analysis**: Analyze your daily routine with advanced language models (FLAN-T5)
- **📊 Region-Aware Calculations**: Precise carbon calculations based on your local electricity grid
- **👥 Community Integration**: Find real solutions from Reddit discussions with AI filtering
- **📈 Progress Tracking**: Visualize your carbon journey over time
- **🌿 Environment AI Assistant**: Ask questions about climate change and sustainability
- **⚙️ Customizable Settings**: Configure location, units, and notifications

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ and pip
- (Optional) Ollama for local LLM support

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Ai Climate Impact Calculator"
   ```

2. **Set up the environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 🏗️ Architecture

### Streamlit Application
- **AI Models**: FLAN-T5 for text analysis and generation
- **Reddit Integration**: Search and analyze community discussions
- **Carbon Calculations**: Region-aware emission factors
- **Interactive UI**: Tabs for different features

### Key Components
- **Calculator Tab**: AI-powered daily routine analysis
- **Analysis Tab**: Advanced region-aware "what-if" optimizer
- **Insights Tab**: Progress tracking and achievements
- **Community Tab**: Reddit search for climate solutions
- **Settings Tab**: User preferences and configuration
- **Environment AI**: Climate Q&A system

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project directory:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

### Optional Data Files

For more accurate calculations, add these CSV files to the `data/` directory:

- `egrid_subregion_factors.csv` - Grid emission factors by region
- `zip_to_egrid.csv` - ZIP code to grid region mapping

## 🎯 Usage

### Calculator Tab
1. Describe your daily routine in natural language
2. Set your carbon footprint goal
3. Get AI-powered analysis and suggestions

### Analysis Tab
1. Enter your location and usage details
2. Try different optimization scenarios
3. View detailed breakdowns and charts

### Community Tab
1. Describe a climate-related problem
2. Find similar stories and solutions from Reddit
3. Get AI-summarized advice

### Environment AI
1. Ask questions about climate change and sustainability
2. Get informative, science-based answers

## 🛠️ Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
streamlit run app.py --server.runOnSave true
```

## 📊 Features in Detail

### AI-Powered Analysis
- Uses FLAN-T5 or Ollama models for natural language processing
- Analyzes daily routines and provides environmental impact assessments
- Generates personalized recommendations

### Region-Aware Calculations
- Considers local electricity grid intensity
- Supports ZIP code-based calculations (US)
- Fallback to regional averages (US, EU, Global)

### Community Integration
- Searches Reddit for real-world solutions
- AI-powered relevance filtering
- Summarizes findings into actionable advice

### Carbon Footprint Tracking
- Tracks progress over time
- Visualizes trends with charts
- Provides achievement system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## �� Acknowledgments

- Built with Streamlit
- AI models powered by Hugging Face Transformers
- Community data from Reddit
- Icons and styling from Streamlit components

---

**Made with ❤️ for the planet**
