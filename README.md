# AI Comment Generator & Analyzer 💬

An interactive Streamlit application that generates realistic Dutch comments using LLAMA 3.2 3B AI model and analyzes their sentiment and similarity to the original post.

## Features ✨

- 🤖 **AI-Powered Comment Generation**: Uses LLAMA 3.2 3B Instruct model to generate realistic Dutch Facebook-style comments
- 📊 **Advanced Sentiment Analysis**: Multi-model ensemble voting system with three BERT-based models for accurate Dutch sentiment detection
- 🎯 **Similarity Metrics**: Calculates cosine similarity between the post and each comment using sentence transformers
- 📈 **Visual Analytics**: Beautiful interactive charts and gauge meters for easy interpretation
- 💾 **Export Functionality**: Download analysis results as CSV for further processing
- 🎨 **Modern UI**: Clean, professional interface with gradient headers and responsive design

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster inference)
- At least 8GB RAM

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd "c:\Users\20245179\OneDrive - TU Eindhoven\WP2_Simulation_Tool"
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data (for sentence tokenization):**
```bash
python -c "import nltk; nltk.download('punkt')"
```

## Usage

### Running the Application

1. **Start the Streamlit app:**
```bash
streamlit run comment_generator_app.py
```

2. **The app will open in your default browser** (usually at http://localhost:8501)

### Using the App

1. **Load Models**: Click "🚀 Load AI Models" in the sidebar (first time may take a few minutes)

2. **Enter Your Post**: Type or paste your Dutch text in the text area

3. **Set Number of Comments**: Choose how many comments to generate (1-10)

4. **Generate**: Click "🎯 Generate Comments" button

5. **View Results**: 
   - See overall statistics (average similarity, sentiment, etc.)
   - View interactive charts showing similarity and sentiment distributions
   - Expand individual comment cards to see detailed gauges
   - Export results as CSV if needed

## How It Works

### Comment Generation
The app uses LLAMA 3.2 3B Instruct model with few-shot prompting:
- Provides example post and comments in Dutch
- Instructs the model to generate realistic Facebook-style responses
- Extracts clean comments using regex pattern matching
- Implements retry logic for robust generation

### Similarity Analysis
Uses multilingual Sentence Transformers:
- Encodes both post and comments using `paraphrase-multilingual-MiniLM-L12-v2`
- Calculates cosine similarity between embeddings
- Displays similarity scores on 0-100% scale with color-coded gauges

### Sentiment Analysis
Employs ensemble voting with multiple BERT-based models:
- **Three specialized models**:
  1. `nlptown/bert-base-multilingual-uncased-sentiment` - Multilingual sentiment (1-5 stars)
  2. `clips/republic` - Dutch-specific sentiment model
  3. `citizenlab/twitter-xlm-roberta-base-sentiment-finetunned` - Social media sentiment
- **Sentence-level analysis**: Tokenizes comments into sentences and analyzes each
- **Majority voting**: Combines predictions from all models using strict majority (≥2 models)
- **Multi-level approach**: 
  - Analyzes each sentence independently
  - Aggregates sentence sentiments for overall comment sentiment
  - Provides both granular and high-level insights
- **Robust classification**: Handles edge cases and mixed sentiments
- **Visual output**: Color-coded meters (🟢 Positive / 🟡 Neutral / 🔴 Negative) with emoji indicators

## Project Structure

```
WP2_Simulation_Tool/
├── comment_generator_app.py    # Main Streamlit application
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Models Used

1. **LLAMA 3.2 3B Instruct** (`meta-llama/Llama-3.2-3B-Instruct`)
   - 8-bit quantization for efficient memory usage
   - Few-shot prompting for Dutch comment generation

2. **Sentiment Analysis Ensemble** (Multi-model voting)
   - `nlptown/bert-base-multilingual-uncased-sentiment` - Multilingual 5-star sentiment
   - `clips/republic` - Dutch-specific sentiment classifier
   - `citizenlab/twitter-xlm-roberta-base-sentiment-finetunned` - Social media sentiment
   - Majority voting system for robust predictions

3. **Sentence Transformers** (`paraphrase-multilingual-MiniLM-L12-v2`)
   - Multilingual support including Dutch
   - Fast and accurate semantic similarity

## Performance Tips

- **GPU Acceleration**: The app works best with a CUDA-compatible GPU
- **Model Caching**: Models are cached after first load using `@st.cache_resource`
- **Memory**: Close other GPU-intensive applications for optimal performance
- **Batch Size**: Generate 1-5 comments at a time for faster results

## Troubleshooting

### Models not loading
- Ensure you have sufficient disk space (models ~5-6GB total including sentiment models)
- Check your internet connection for first-time download
- Verify CUDA installation if using GPU
- Sentiment models may take 2-3 minutes to load initially

### Out of Memory errors
- Try using a smaller number of comments
- Close other applications
- Consider using CPU-only mode (remove `load_in_8bit=True`)

### Slow generation
- First generation is always slower (model loading)
- Subsequent generations use cached models
- GPU significantly speeds up generation

## Export Format

CSV export includes:
- `Comment`: The generated comment text
- `Similarity`: Cosine similarity score (0-1)
- `Sentiment_Score`: Sentiment score (0-100 scale)
- `Sentiment_Label`: Text label with emoji (Positive 😊 / Neutral 😐 / Negative 😞)
- `Sentiment_Numeric`: Numeric sentiment (-1, 0, or 1)
- `Model_Predictions`: Individual predictions from all sentiment models
- `Sentence_Sentiments`: Sentiment breakdown per sentence

## Requirements

See `requirements.txt` for full dependency list. Key packages:
- streamlit: Web interface
- transformers: LLAMA model and sentiment models
- sentence-transformers: Similarity analysis
- plotly: Interactive visualizations
- nltk: Sentence tokenization for sentiment analysis
- torch: Deep learning backend

## License

This project is for research and educational purposes.

## Credits

- LLAMA model by Meta AI
- Sentence Transformers by UKPLab
- Built with Streamlit
