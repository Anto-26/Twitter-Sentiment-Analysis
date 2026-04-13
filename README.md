# Twitter Sentiment Analysis (NLP Machine Learning Project)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/ML-NLP-orange)
![Status](https://img.shields.io/badge/Status-Active-green)
![Dataset](https://img.shields.io/badge/Dataset-Sentiment140-lightgrey)

## Overview

This project performs sentiment analysis on tweets using the Sentiment140 dataset.

The dataset contains 1.6 million tweets labeled as:
0 = Negative sentiment
4 = Positive sentiment

The goal is to build a machine learning model that can classify tweet sentiment accurately using NLP techniques.

## Project Structure

Twitter-Sentiment-Analysis/

notebooks/              Jupyter notebooks for EDA and experiments  
src/                   
src/dataloading/              Data handling utilities  
src/preprocessing/     Text preprocessing scripts  
src/models/             Model training scripts  

data/                  Dataset (not pushed to GitHub)  
requirements.txt  
README.md  

## Dataset

The Sentiment140 dataset contains:
1.6 million tweets collected using the Twitter API

Columns:
sentiment
ids
date
flag
user
text

## Machine Learning Pipeline

Raw tweets
Text cleaning (removal of URLs, mentions, punctuation)
Tokenization and normalization
Feature extraction using TF-IDF or CountVectorizer
Model training using classification algorithms
Model evaluation

## Installation

Clone the repository:
git clone https://github.com/Anto-26/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis

Create virtual environment:
python -m venv venv
source venv/bin/activate   (Mac/Linux)
venv\Scripts\activate      (Windows)

Install dependencies:
pip install -r requirements.txt

## Usage

Run preprocessing script:
python src/preprocessing/preprocess.py

Run Jupyter notebook:
jupyter notebook

## Results

Model performance (approximate):

Logistic Regression: 80–85% accuracy

Results may vary depending on preprocessing and feature engineering.

## Notes

Dataset file (Sentiment140.csv) is not included due to large size.
Place dataset inside the data folder locally.

Virtual environment (venv/) is ignored using .gitignore.

## Author

James Arnold

## License

This project is licensed under the MIT License.