# 🧠 Gender Classification NLP (Flask App)

## 🔍 Overview

This project predicts the **gender of an author** based on text input using **Natural Language Processing (NLP)** and **Machine Learning**.
It includes a **Flask web interface** for real-time predictions and is deployable using **Gunicorn**.

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

## 🚀 How to Run

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
