# 🧠 Stress Detection from Social Media Posts

## Project Summary

**Stress Detection using Machine Learning** is an NLP-based project that analyzes social media text and classifies it as **Stress** or **No Stress**. The goal is to demonstrate how machine learning can be applied to real-world mental health–related text data to identify early signs of psychological stress.

This project showcases end-to-end skills in **data preprocessing, NLP, feature engineering, model training, and evaluation**.

---

## Problem Statement

People often express stress, anxiety, and emotional struggles through social media posts. Manually identifying these signals at scale is difficult.

This project answers the question:

> *Can machine learning automatically detect stress from short text posts shared online?*

---

## Dataset

* Source: Mental health–related Reddit posts (Kaggle)
* Total columns: 116
* Columns used:

  * **text** – user-generated social media content
  * **label** – stress indicator

    * `0` → No Stress
    * `1` → Stress

The dataset is fully labeled and contains no missing values.

---

## Data Preprocessing & NLP Pipeline

To improve model performance, the following steps were applied:

* Lowercasing text
* Removing URLs, punctuation, numbers, and HTML tags
* Stopword removal using NLTK
* Stemming with Snowball Stemmer

A **Word Cloud visualization** was used to explore commonly used words in mental health discussions.

---

## Model Development

* **Feature Extraction:** Bag of Words (CountVectorizer)
* **Algorithm:** Bernoulli Naive Bayes
* **Task:** Binary text classification
* **Train/Test Split:** 67% / 33%

Bernoulli Naive Bayes was selected due to its effectiveness in binary classification problems involving text data.

---

## Sample Predictions

**Input:**

> People need to take care of their mental health

**Prediction:**

> No Stress

**Input:**

> Sometimes I feel like I need some help

**Prediction:**

> Stress

---

## Tools & Technologies

* Python
* Pandas & NumPy
* NLTK
* Scikit-learn
* Matplotlib
* WordCloud

---

## Key Learnings

* Text preprocessing has a major impact on model performance
* Even simple models can produce strong results with clean data
* NLP pipelines are highly effective for social media text analysis
* Binary classification is well-suited for mental health screening tasks

---

## Potential Improvements

* Replace Bag of Words with **TF-IDF**
* Experiment with Logistic Regression, SVM, or Random Forest
* Use transformer models like **BERT** for deeper semantic understanding
* Deploy as a web app or REST API

---

## Ethical Considerations

This project is for **educational and research purposes only**. It is **not a medical diagnostic tool** and should not replace professional mental health support.

---

## Project Links

* **GitHub Repository:** *[Solomon Adegoke](https://www.linkedin.com/in/solomon-eniola-adegoke)*
* **Dataset:** Kaggle – Mental Health Reddit Dataset

---

## Author

**Solomon Adegoke**
Data Scientist - AI/ML Engineer

---

📌 *This project demonstrates practical NLP and machine learning skills applied to a real-world social problem.*
