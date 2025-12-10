# BBC-News-Summary-ML-DL-Project

A media monitoring project that classifies BBC news articles, discovers latent topics, and builds a reinforcement learning (RL) agent to decide whether to use a classical ML model, a deep learning model, or escalate a news item to humans.

> **Author:** Ainedembe Denis  

---

## 1. Project Overview

A media monitoring startup needs an intelligent system that can:

1. **Classify news articles** into categories.
2. **Group articles into topics** using clustering.
3. **Use reinforcement learning** to decide:
   - When to trust a classical ML classifier,
   - When to use a deep learning model,
   - When to escalate difficult cases to human reviewers.

This repository implements the full pipeline using the **BBC News Summary dataset** from Kaggle https://www.kaggle.com/datasets/pariza/bbc-news-summary

---

## 2. Dataset

- **Name:** BBC News Summary  
- **Source:** Kaggle – BBC News Summary Dataset  
- **Contents:** Short news summaries and categories (e.g., business, entertainment, politics, sport, tech).

> You must first download the dataset manually from Kaggle and place it in the `dataset/` folder.

Suggested structure:

```bash
BBC-News-Summary-ML-DL-Project/
├─ dataset/  # raw dataset from Kaggle
├─ bbc-news-summary.ipynb
├─ reports/
└─ README.md
