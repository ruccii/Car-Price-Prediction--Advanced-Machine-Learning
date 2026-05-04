# 🚗 Car Price Prediction - Advanced Machine Learning

**A robust car price prediction system** built with advanced Machine learning techniques, achieving **92.7% R² accuracy**.

This project demonstrates a complete end-to-end machine learning pipeline — from exploratory data analysis and sophisticated feature engineering to ensemble modeling, model interpretability with SHAP, and a production-ready web application.

![Banner](assets/sell-car-hero-banner.png)

## ✨ Features

- **Comprehensive EDA** with insightful visualizations
- **Advanced Data Cleaning** & Feature Engineering (including derived features like vehicle age, condition, and usage)
- **Dimensionality Reduction** — PCA & t-SNE
- **Clustering Analysis** — K-Means
- **Feature Selection** — Recursive Feature Elimination (RFE)
- **Multiple Models**:
  - Random Forest
  - Gradient Boosting
  - Stacking Ensemble
- **Model Interpretability** using **SHAP** values
- **Interactive Web App** built with Streamlit for instant price predictions

## 📊 Project Highlights

- **Best Performance**: 92.7% R² Score
- **Technologies**: Python, scikit-learn, pandas, numpy, matplotlib/seaborn, SHAP, Streamlit
- **Deployment Ready**: Streamlit app included (`car_app.py`)

## 🛠️ Tech Stack

| Component              | Technologies                          |
|------------------------|---------------------------------------|
| Language               | Python 3                              |
| Data Analysis          | pandas, numpy                         |
| Visualization          | matplotlib, seaborn, plotly           |
| ML Models              | scikit-learn (RF, GB, Stacking)       |
| Interpretability       | SHAP                                  |
| Web App                | Streamlit                             |
| Others                 | Joblib (model persistence)            |

## 📁 Repository Structure

```bash
├── car price prediction.ipynb          # Main ML pipeline & analysis
├── Advanced MLC analysis.ipynb         # Advanced techniques (PCA, t-SNE, Clustering, RFE)
├── car_app.py                          # Streamlit web application
├── requirements.txt                    # Project dependencies
├── assets/                             # Images & banners
├── docs/                               # Reports & documentation
│   ├── Analysis Report .pdf
│   └── Car price prediction report.pdf
└── README.md
```

The goal was to build a data driven solution that can assist buyers, sellers, and businesses in making informed pricing decisions.
