# Sentiment Analysis on IMDB Reviews 🎬

## Overview
This project classifies movie reviews as **positive** or **negative** using natural language processing and machine learning.

- Text cleaning: lowercasing, removing punctuation, stopwords
- TF-IDF vectorization
- Logistic Regression classifier
- Saved model and vectorizer for easy prediction


## How to Run
1. Clone the repository
```bash
git clone https://github.com/A1Sharif/MLProject.git
cd MLProject
```

2. Create and activate the virtual environment
```bash

python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```


3. Install the dependencies
```bash
pip install -r requirements.txt
```

## USAGE

1. Train the model
```bash
python src/train_model.py
```

2. Predict the sentiment
```bash
python src/predict.py
```
