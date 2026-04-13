# 📊 Twitter Sentiment Analysis (NLP Machine Learning Project)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/ML-NLP-orange)
![Status](https://img.shields.io/badge/Status-Active-green)
![Dataset](https://img.shields.io/badge/Dataset-Sentiment140-lightgrey)

---

## 📌 Overview

This project builds a **machine learning model for sentiment analysis on tweets** using the **Sentiment140 dataset**.

The model classifies tweets into:
-  **Negative (0)**
-  **Positive (4)**

It demonstrates a full **NLP pipeline** including data preprocessing, feature extraction, and model training.


## 📂 Project Structure
Twitter-Sentiment-Analysis/
│
├── notebooks/ # Exploratory Data Analysis & experiments
│ └── EDA.ipynb
│
├── src/
│ ├── dataloading/ # Data handling utilities
│ ├── preprocessing/ # Text cleaning scripts
│ └── models/ # ML model training scripts
│ └── features/ #TFIDF Vectoriser
│
├── data/ # Dataset (ignored in Git)
├── requirements.txt
└── README.md

## 📊 Dataset

We use the **Sentiment140 dataset**, which contains:

- 1.6 million tweets
- Extracted using Twitter API
- Columns:
  - sentiment (0 = negative, 4 = positive)
  - ids
  - date
  - flag
  - user
  - text

---

## ⚙️ ML Pipeline

The workflow followed in this project:
Raw Tweets
↓
Text Cleaning (URLs, mentions, punctuation removal)
↓
Feature Extraction (TF-IDF)
↓
Model Training (Logistic Regression / ML models)
↓
Evaluation