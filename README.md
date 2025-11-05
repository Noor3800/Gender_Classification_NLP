# 🧠 Gender Classification NLP (Flask App)

## 🔍 Overview

A **Flask-based NLP web app** that predicts the **gender of an author** based on written text.
The model uses a **TF-IDF vectorizer** and a **trained machine learning classifier**, integrated into an interactive web interface for real-time predictions.


## 📦 Files

```
├── app.py              # Main Flask app
├── model.pkl           # Trained classification model
├── tfidf.pkl           # TF-IDF vectorizer
├── templates/
│   └── index.html      # Webpage template
├── requirements.txt    # Dependencies
└── Procfile            # For deployment (Gunicorn)
```
---

## ⚙️ Tech Stack

* **Python**, **Flask**
* **Scikit-learn**, **NLTK**, **Pickle**
* **Pandas**, **NumPy**, **Regex**
* **Gunicorn** (for deployment)

---

## 🧩 Features

* User can enter text to predict gender (Male/Female).
* Pretrained ML model (e.g., Logistic Regression / SVM) loaded via Pickle.
* Text preprocessing with tokenization, stopword removal, and lemmatization using **NLTK**.
* Interactive web interface built with **Flask**.

---
## 🚀 How to Run Locally

### Clone the Repository

```bash
git clone https://github.com/Noor3800/Gender_Classification_NLP.git
cd Gender_Classification_NLP
```

### Install requirements

```bash
pip install -r requirements.txt
```

###  Download NLTK resources

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

###  Run the Flask App

```bash
python app.py
```

The app will start on:

```
http://127.0.0.1:5000/
```

open in browser and enter a sentence, click **Predict**, and see whether the model identifies it as **Male** or **Female**!
---

## 📊 Model Workflow

1. **Preprocessing:** Clean text (lowercasing, removing punctuation, stopwords, and lemmatization).
2. **Vectorization:** Convert text into numerical form using TF-IDF.
3. **Model Prediction:** Predict gender label based on linguistic patterns.

---

## Example

**Input:**

> “I love coding and exploring new technologies!”

**Output:**

> Predicted Gender: *Male / Female*

---

## Skills Practiced

* NLP text preprocessing and feature extraction
* Flask app development and API integration
* Model deployment using Gunicorn
* Data handling and machine learning

---
