# 📰 Fake News Detector

A Streamlit web application that helps students and users analyze news articles, assess their credibility, and generate summaries to prevent misinformation.

## 🌟 Features

- **Fake News Detection**: Analyze articles and get confidence scores for fake vs real classification
- **Article Summarization**: Generate concise summaries of news articles
- **Analysis History**: Track and view all your previous analyses
- **Export Functionality**: Download results as JSON or CSV files
- **User-Friendly Interface**: Clean, intuitive design perfect for students
- **Real-time Analysis**: Get instant results with loading indicators
 - **Sidebar Status (Non-ML)**: See session stats and model load status
 - **Input Enhancements (Non-ML)**: Live character counter and max length limit
 - **Duplicate Prevention (Non-ML)**: Warns on re-analyzing identical text
 - **Cooldown (Non-ML)**: Short cooldown between analyses to avoid accidental spamming
 - **Clear Input (Non-ML)**: One-click button to clear the text area

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd Fack_detector
   ```

2. **Install required packages**
   ```bash
   pip install streamlit transformers torch
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, copy the URL from your terminal

## 🤖 AI Models Used

### Summarization Model
- **Model**: `sshleifer/distilbart-cnn-12-6`
- **Purpose**: Creates concise summaries of news articles (30-120 characters)
- **Type**: Distilled BART model optimized for summarization

### Fake News Detection Model
- **Model**: `mrm8488/bert-tiny-finetuned-fake-news-detection`
- **Purpose**: Classifies articles as fake or real with confidence scores
- **Type**: Fine-tuned BERT model specifically trained for fake news detection

## 📖 How to Use

1. **Paste Article**: Copy and paste a news article or paragraph into the text area
2. **Analyze**: Click the "Analyze Article" button
3. **View Results**: 
   - Read the generated summary
   - Check the credibility assessment with confidence percentage
   - View detailed probability scores for fake vs real classification
4. **Download**: Save individual results or entire analysis history
5. **Track History**: View all your previous analyses in the history section
6. **Use Sidebar**: Monitor analyses count, last result time, and tweak max input length

## 📊 Output Information

### Summary
- Concise overview of the article content
- Optimized for readability and comprehension

### Credibility Check
- **FAKE**: Article appears to be fake news (with confidence %)
- **REAL**: Article appears to be legitimate news (with confidence %)
- **UNKNOWN**: Credibility is unclear

### Detailed Probabilities
- Shows exact percentages for both fake and real classifications
- Provides transparency in the AI's decision-making process

## 💾 Data Management

### Analysis History
- All analyses are stored in your browser session
- History includes timestamps, predictions, confidence scores, and summaries
- View history in a clean table format

### Export Options
- **Individual Results**: Download single analysis as JSON
- **Full History**: Export all analyses as JSON or CSV
- **File Naming**: Automatic timestamp-based naming for easy organization

### Data Storage
- **Session-based**: History persists during your browser session
- **Local Storage**: No data sent to external servers
- **Privacy-Focused**: All analysis happens locally on your machine

## 🛠️ Technical Details

### Framework
- **Streamlit**: Web application framework
- **Transformers**: Hugging Face library for AI models
- **Python**: Backend programming language

### Performance
- **Model Caching**: Models load once per session for faster analysis
- **Memory Efficient**: Uses optimized BERT-tiny model for speed
- **Real-time Processing**: Instant analysis results

### Dependencies
```
streamlit
transformers
torch
```

## 📁 Project Structure

```
Fack_detector/
├── app.py                 # Main application file
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── README.md             # This file
└── requirements.txt      # Python dependencies (optional)
```

## 🔧 Configuration

The app uses default Streamlit settings. You can customize the appearance by modifying `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 🚨 Important Notes

### Model Limitations
- Models are trained on specific datasets and may not catch all types of misinformation
- Results should be used as a guide, not absolute truth
- Always verify important information through multiple reliable sources

### Data Privacy
- All processing happens locally on your machine
- No data is sent to external servers
- Analysis history is stored only in your browser session

### System Requirements
- Requires internet connection for initial model download
- Models will be cached locally after first use
- Recommended: 4GB+ RAM for optimal performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Troubleshooting

### Common Issues

**Model Loading Failed**
- Ensure you have a stable internet connection
- Try refreshing the page
- Check if you have sufficient disk space for model downloads

**App Won't Start**
- Verify Python version (3.7+)
- Install all required dependencies
- Check if port 8501 is available

**Slow Performance**
- Close other browser tabs
- Restart the application
- Ensure sufficient RAM is available

### Getting Help

If you encounter issues:
1. Check the error messages in the app
2. Verify your Python environment
3. Ensure all dependencies are installed correctly
4. Create an issue in the repository

## 🙏 Acknowledgments

- **Hugging Face**: For providing the AI models and transformers library
- **Streamlit**: For the excellent web framework
- **Model Authors**: For training and sharing the detection models

---

**Made with ❤️ for students and educators to combat misinformation**
