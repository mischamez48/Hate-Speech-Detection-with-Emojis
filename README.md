# Hate Speech Detection and Emoji Analysis - Deep Learning Project

## Project Overview

This project implements a comprehensive ensemble approach for hate speech detection using multiple datasets and deep learning models. The work was completed as part of the EE559 course at EPFL and focuses on analyzing text content with emoji processing and hate speech classification.

## Team

**Group 26:**
- Project ID: 310752, 316483, 316146
- Course: EE559 - Deep Learning (EPFL)

## Project Structure

```
├── hatemoji_ensemble.ipynb     # Main ensemble model implementation
├── model_for_tdavidson.ipynb   # Davidson dataset model
├── model_for_ethos.ipynb       # ETHOS dataset model  
├── model_for_emoji.ipynb       # Emoji-specific model
├── EE_559_Group26_Rapport.pdf  # Project report (French)
├── EE559_Group26_Poster.pdf    # Project poster
└── README.md                   # This file
```

## Key Features

### 🎯 Multi-Dataset Approach
- **Davidson Dataset**: Hate speech and offensive language detection
- **ETHOS Dataset**: Multilingual hate speech detection
- **HatemojiCheck Dataset**: Emoji-based hate speech analysis
- **Custom Emoji Processing**: Advanced emoji-to-text conversion

### 🧠 Model Architecture
- **DistilBERT-based Models**: Fine-tuned for each specific dataset
- **Ensemble Learning**: Combines multiple specialized models
- **Text Preprocessing**: Advanced preprocessing including emoji handling
- **Multi-class Classification**: Hate speech, offensive language, and neutral content

### 🔧 Technical Implementation
- **Framework**: PyTorch with Transformers library
- **Tokenization**: DistilBERT tokenizer with custom preprocessing
- **Training**: Custom training loops with validation
- **Evaluation**: Comprehensive metrics and analysis

## Datasets Used

1. **HatemojiCheck** (`HannahRoseKirk/HatemojiCheck`)
   - Focus on emoji-based hate speech detection
   - Includes various test groups and functionality checks

2. **Davidson Hate Speech** (`tdavidson/hate_speech_offensive`)
   - Classic hate speech detection dataset
   - Three-class classification: hate speech, offensive language, neither

3. **ETHOS Dataset**
   - Multilingual hate speech detection
   - Binary and multi-label classification tasks

## Key Preprocessing Features

### Emoji Processing
```python
def process_text_with_emojis(text):
    # Convert emojis to descriptive text
    # Handle spacing and formatting
    # Clean and normalize output
```

### Text Normalization
- **Leet speak conversion**: (4→a, 3→e, 1→i, 0→o)
- **Social media cleaning**: Remove @mentions, RT, hashtags
- **URL removal**: Clean web links
- **Character normalization**: Handle repeated characters

## Model Architectures

### DistilBERT Classifier
```python
class DistilBERTClassifierDavidsonDataset(PreTrainedModel):
    def __init__(self, config, weights=None):
        # DistilBERT backbone
        # Custom classification layers
        # Dropout regularization
```

### Training Configuration
- **Batch Size**: 64 (optimized for hardware)
- **Learning Rate**: Adaptive with scheduler
- **Validation Strategy**: Stratified train/validation/test splits
- **Hardware Optimization**: Thread limiting for JupyterHub EPFL

## Results and Performance

The project achieved successful implementation across multiple datasets with:
- Comprehensive evaluation metrics
- Ensemble model performance analysis
- Detailed error analysis and insights
- Visual performance comparisons

*Detailed results are available in the project report (`EE_559_Group26_Rapport.pdf`)*

## Installation and Usage

### Prerequisites
```bash
pip install torch transformers datasets emoji preprocessor
pip install pandas matplotlib numpy scikit-learn
```

### Running the Models

1. **Main Ensemble Model**:
   ```bash
   jupyter notebook hatemoji_ensemble.ipynb
   ```

2. **Individual Dataset Models**:
   ```bash
   jupyter notebook model_for_tdavidson.ipynb
   jupyter notebook model_for_ethos.ipynb
   jupyter notebook model_for_emoji.ipynb
   ```

### Hardware Requirements
- GPU recommended for training
- Optimized for EPFL JupyterHub environment
- CPU training supported with thread optimization

## Key Contributions

1. **Novel Emoji Processing**: Advanced emoji-to-text conversion methodology
2. **Multi-Dataset Ensemble**: Combining insights from diverse hate speech datasets
3. **Preprocessing Pipeline**: Comprehensive text cleaning and normalization
4. **Performance Analysis**: Detailed evaluation across different model configurations

## Technical Highlights

- **Advanced Text Preprocessing**: Custom functions for social media text
- **Emoji Handling**: Sophisticated emoji-to-text conversion
- **Model Fine-tuning**: DistilBERT adaptation for hate speech detection
- **Ensemble Methods**: Combining multiple specialized models
- **Performance Optimization**: Hardware-aware training configuration

## Future Work

- Extension to more languages and datasets
- Real-time deployment capabilities
- Advanced ensemble techniques
- Improved emoji semantic understanding

## References

- Davidson, T., et al. "Hate Speech Detection with a Computational Approach"
- Kirk, H.R., et al. "HatemojiCheck: A Test Suite for Content Moderation"
- ETHOS: Online Hate Speech Detection Dataset
- DistilBERT: A distilled version of BERT

## License

This project was developed for academic purposes as part of the EE559 Deep Learning course at EPFL.

## Contact

For questions about this implementation, please refer to the detailed project report or contact the course instructors.

---

*This project demonstrates advanced techniques in natural language processing, hate speech detection, and ensemble learning methods using state-of-the-art transformer models.* 