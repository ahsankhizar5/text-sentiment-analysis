# 🎭 Text Sentiment Analysis

A machine learning pipeline to classify IMDB movie reviews as **positive** or **negative** using NLP preprocessing, TF-IDF vectorization, and a Logistic Regression model.

---

## 🧠 Overview

This project builds a **text sentiment analysis** model using the [IMDB Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The pipeline involves preprocessing, vectorization, training, evaluation, and live prediction.

---

## 📁 Dataset

This project uses the **IMDB Reviews Dataset**:

➡️ [Download from Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Steps:
1. Download and extract the dataset.
2. Rename or ensure the file is named `IMDB_Dataset.csv`.
3. Place it in the project root directory.

> ⚠️ The dataset is not included in the repo due to GitHub file size limits.

---

## 🛠️ Tech Stack

- Python 3
- Pandas
- Scikit-learn
- NLTK
- TF-IDF Vectorizer
- Logistic Regression

---

## 🚀 Features

- Clean and normalize text using NLP techniques
- Convert reviews into numerical features using TF-IDF
- Train and evaluate a logistic regression model
- Save trained model and vectorizer for reuse
- Predict sentiment of custom reviews in real-time

---

## 📊 Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 85.13% |
| F1-Score  | 85%   |

---

## 📦 Project Structure

```

text-sentiment-analysis/
├── IMDB\_Dataset.csv
├── sentiment\_analysis.py
├── sentiment\_model.pkl
├── tfidf\_vectorizer.pkl
└── README.md

````

---

## ▶️ Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/ahsankhizar5/text-sentiment-analysis.git
   cd text-sentiment-analysis
````

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the script

   ```bash
   python sentiment_analysis.py
   ```

4. Enter your own review for live prediction!

---

## 📌 Example

```text
📝 Try your own review:
Enter a movie review: this seems to be bad one
Predicted Sentiment: Negative 😞
```

---

## 📑 License

MIT License

---

## 🤝 Contact

For queries or collaboration, feel free to reach out:
**Ahsan Khizar**
[GitHub](https://github.com/ahsankhizar5) — [LinkedIn](https://linkedin.com/in/ahsankhizar5)

> “Code is not just about solving problems. It’s about building trust, clarity, and real-world impact — one line at a time.”> — *Ahsan Khizar*
