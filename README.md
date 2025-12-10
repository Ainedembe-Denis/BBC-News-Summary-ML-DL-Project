# BBC-News-Summary-ML-DL-Project

A media monitoring startup needs an intelligent system that can **classify news**, **group articles into topics**, and **decide when to escalate to humans** using **reinforcement learning**.  
This project uses the **BBC News Summary** dataset to build:

- A classical ML news classifier  
- A deep learning news classifier  
- A topic clustering pipeline  
- A Q-learning RL agent that chooses between ML, DL, or human escalation

> **Author:** Ainedembe Denis  
> **Dataset:** [BBC News Summary (Kaggle)](https://www.kaggle.com/datasets/pariza/bbc-news-summary)  
> **Purpose:** Academic assignment / educational use

---

## 1. Dataset Description

The project uses the **BBC News Summary** dataset, which contains:

- **News Articles:** 417 political news articles from 2004–2005 in the `News Articles` folder  
- **Summaries:** Five extractive summaries per article in the `Summaries` folder  
- The **first clause** of each article text is the title  

The dataset was originally derived from a broader BBC news corpus used for document clustering and categorization (Greene & Cunningham, ICML 2006).

In this project, we primarily treat each **news article** as a text document with an associated **category/label**, suitable for classification and clustering tasks.

---

## 2. Project Objectives (Assignment Breakdown)

### Part A – Data Mining and Preprocessing

1. Load the BBC dataset from local storage.
2. Display example articles and the class distribution.
3. Clean text:
   - Lowercasing
   - Removing punctuation and HTML tags
4. Prepare:
   - **TF-IDF vectors** for classical ML models  
   - **Tokenized + padded sequences** for deep learning models

---

### Part B – Two News Classifiers

Build and compare two models:

1. **Classical ML model**  
   - Logistic Regression / SVM / Naive Bayes using TF-IDF features  

2. **Deep Learning model**  
   - LSTM / CNN / BERT (choose at least one)

The notebook must output:

- F1 scores **per class**
- Confusion matrix
- **5 misclassified examples** with their true vs predicted labels
- Deep learning **training curves** (loss & accuracy vs epochs)

---

### Part C – Topic Clustering

1. Convert news articles to **embeddings** using one of:
   - TF-IDF  
   - Doc2Vec  
   - BERT (or similar transformer embeddings)
2. Apply **k-Means** or **Agglomerative Clustering**
3. Visualize clusters using **PCA** or **t-SNE**
4. Extract **top keywords per cluster** to interpret the topics

---

### Part D – RL Decision Agent (15 marks)

Design a **Q-learning agent** that chooses how to handle each article:

- **State representation** includes:
  - ML model confidence
  - DL model confidence
  - Cluster ID
  - Article length
  - Disagreement flag (ML prediction ≠ DL prediction)

- **Actions:**
  - `0` = Use classical ML prediction  
  - `1` = Use deep learning prediction  
  - `2` = Escalate to human

- Define and justify a **reward function**, e.g.:
  - Positive reward for correct automated decisions
  - Penalty for incorrect predictions
  - Small penalty or smaller reward for escalation

- Train Q-learning for **1000–1500 episodes**

Outputs:

- Reward curve across episodes
- Action distribution (how often ML, DL, and escalation are chosen)
- Accuracy comparison: **ML vs DL vs RL agent**
- Final learned **Q-table**

---

## 3. Tech Stack

Planned tools and libraries (in the notebook):

- **Python 3.x**
- **Data handling:** `pandas`, `numpy`
- **Preprocessing & ML:** `scikit-learn`
  - TF-IDF, train–test split, metrics, clustering, PCA/t-SNE
- **Deep Learning:** `tensorflow` / `keras` or `PyTorch` (e.g., LSTM/CNN)
- **NLP utilities:** `nltk` / `spaCy` (tokenization, stopwords)
- **Visualization:** `matplotlib`, `seaborn`
- **RL (Q-learning):** custom implementation with `numpy`

Project structure:

```bash
BBC-News-Summary-ML-DL-Project/
├─ dataset/  # raw dataset from Kaggle
│  ├─ News Articles/        # raw BBC articles
│  └─ Summaries/            # extractive summaries
├─ bbc_news_ml_dl_rl.ipynb
├─ results/
│  ├─ screenshots/      
│  └─ reports/     
└─ README.md

