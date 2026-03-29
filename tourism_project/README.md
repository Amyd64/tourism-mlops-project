# 🏖️ Tourism Package Prediction — MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow?logo=huggingface)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green?logo=githubactions)

---

## 📌 Business Context

**"Visit with Us"** is a leading travel company introducing a new **Wellness Tourism Package**. The challenge is identifying which customers are likely to purchase the package before contacting them. The manual approach is inconsistent and time-consuming.

This project builds an automated **end-to-end MLOps pipeline** that predicts customer purchase likelihood, enabling the sales team to target the right customers efficiently.

---

## 🎯 Objective

Design and deploy an MLOps pipeline on GitHub to automate the complete machine learning workflow:
- Data registration and preparation
- Model training with hyperparameter tuning and experiment tracking
- Model deployment as a Streamlit web application
- CI/CD automation using GitHub Actions

---

## 📁 Project Structure

```
tourism-mlops-project/
│
├── tourism_project/
│   ├── data/
│   │   └── tourism.csv                     # Raw dataset
│   │
│   ├── model_building/
│   │   ├── data_register.py                # Upload data to Hugging Face
│   │   ├── prep.py                         # Data cleaning, encoding, train/test split
│   │   └── train.py                        # Model training, MLflow tracking, model upload
│   │
│   ├── deployment/
│   │   ├── app.py                          # Streamlit web application
│   │   ├── Dockerfile                      # Docker container configuration
│   │   └── requirements.txt               # App dependencies
│   │
│   ├── hosting/
│   │   └── hosting.py                      # Push deployment files to HF Space
│   │
│   └── requirements.txt                   # Pipeline dependencies
│
└── .github/
    └── workflows/
        └── pipeline.yml                   # GitHub Actions CI/CD workflow
```

---

## 🗂️ Dataset

The dataset contains **4,128 customer records** with the following features:

| Feature | Description |
|---|---|
| `CustomerID` | Unique identifier (dropped during preprocessing) |
| `ProdTaken` | **Target** — whether customer purchased (0=No, 1=Yes) |
| `Age` | Age of the customer |
| `TypeofContact` | How the customer was contacted |
| `CityTier` | City category (Tier 1 > Tier 2 > Tier 3) |
| `Occupation` | Customer's occupation |
| `Gender` | Gender of the customer |
| `MonthlyIncome` | Gross monthly income |
| `DurationOfPitch` | Duration of the sales pitch (minutes) |
| `NumberOfFollowups` | Number of follow-ups by salesperson |
| `PitchSatisfactionScore` | Customer's satisfaction with the pitch |
| `Passport` | Whether the customer holds a passport |
| `NumberOfTrips` | Average trips taken annually |

---

## ⚙️ MLOps Pipeline

The pipeline runs automatically on every push to the `main` branch via **GitHub Actions**:

```
Raw Data (tourism.csv)
        ↓
[Job 1] Data Registration → Upload to Hugging Face Dataset
        ↓
[Job 2] Data Preparation  → Clean + Encode + Split → Upload Train/Test to HF
        ↓
[Job 3] Model Training    → XGBoost + GridSearchCV + MLflow Tracking → Upload to HF Model Hub
        ↓
[Job 4] Deploy & Hosting  → Push Streamlit App to Hugging Face Space
```

---

## 🤖 Model

- **Algorithm:** XGBoost Classifier
- **Tuning:** GridSearchCV with 3-fold cross-validation
- **Tracking:** MLflow (experiment name: `mlops-training-experiment`)
- **Metrics Logged:** Accuracy, F1 Score, AUC-ROC

### Hyperparameter Grid
| Parameter | Values |
|---|---|
| `n_estimators` | 50, 100, 150 |
| `max_depth` | 3, 5, 7 |
| `learning_rate` | 0.01, 0.05, 0.1 |
| `subsample` | 0.7, 0.8, 1.0 |
| `colsample_bytree` | 0.7, 0.8, 1.0 |
| `reg_lambda` | 0.1, 1, 10 |

---

## 🚀 Deployment

The model is deployed as a **Streamlit web app** hosted on **Hugging Face Spaces** using Docker.

The app allows the sales team to:
- Enter customer details (age, income, occupation, pitch info, etc.)
- Click **"Predict"** to instantly see if the customer is likely to purchase
- View the **probability score** to prioritize outreach

🔗 **Live App:** [https://huggingface.co/spaces/Amyd64/tourism-package-predictor](https://huggingface.co/spaces/Amyd64/tourism-package-predictor)

---

## 🔗 Hugging Face Resources

| Resource | Link |
|---|---|
| 📦 Dataset | [Amyd64/tourism-package-prediction](https://huggingface.co/datasets/Amyd64/tourism-package-prediction) |
| 🤖 Model | [Amyd64/tourism-package-model](https://huggingface.co/Amyd64/tourism-package-model) |
| 🌐 Space (App) | [Amyd64/tourism-package-predictor](https://huggingface.co/spaces/Amyd64/tourism-package-predictor) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9 | Core language |
| XGBoost | ML model |
| Scikit-learn | Preprocessing & evaluation |
| MLflow | Experiment tracking |
| Hugging Face Hub | Dataset, model & app hosting |
| Streamlit | Web application UI |
| Docker | App containerization |
| GitHub Actions | CI/CD automation |
| Pandas / NumPy | Data manipulation |

---

## 📋 Requirements

### Pipeline (`tourism_project/requirements.txt`)
```
huggingface_hub==0.32.6
datasets==3.6.0
pandas==2.2.2
scikit-learn==1.6.0
xgboost==2.1.4
mlflow==3.0.1
```

### App (`tourism_project/deployment/requirements.txt`)
```
pandas==2.2.2
huggingface_hub==0.32.6
streamlit==1.43.2
joblib==1.5.1
scikit-learn==1.6.0
xgboost==2.1.4
mlflow==3.0.1
```

---

## 💡 Key Insights

- Customers with **higher monthly income** are more likely to purchase the package
- **Pitch duration** and **number of follow-ups** strongly influence purchase decisions
- Customers from **Tier 1 cities** show higher purchase rates
- Customers who **hold a passport** are significantly more likely to buy a travel package
- **Salaried professionals** and **managers/AVPs** represent the highest converting segments

---

## 👤 Author

**Amyd64**
- GitHub: [github.com/Amyd64](https://github.com/Amyd64)
- Hugging Face: [huggingface.co/Amyd64](https://huggingface.co/Amyd64)

---

<p align="center"><font size=4 color="navyblue"><b>Power Ahead! 🚀</b></font></p>
