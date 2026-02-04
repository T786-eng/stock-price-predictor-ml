# Stock Price Predictor (ML)

A high-performance Python script designed to predict stock prices using **Machine Learning**. This script is optimized for datasets from Kaggle or Yahoo Finance and runs directly in the **VS Code** standard editor.

---

## 🛠 Features
* **Auto-Dataset Detection:** Automatically scans the current folder and uses the first `.csv` file it finds.
* **Feature Engineering:** Automatically generates technical indicators including **10-day MA**, **50-day MA**, and **RSI (Relative Strength Index)**.
* **Robust Preprocessing:** Handles timezone-aware dates, missing values (forward fill), and data sorting.
* **Optimized Training:** Uses a **Random Forest Regressor** with a sampling feature for fast execution on large datasets.



---

## 🚀 Quick Start in VS Code

### 1. Install Dependencies
Open your VS Code terminal (**Ctrl + `**) and run the following command to install the required libraries:
```bash
   pip install pandas numpy scikit-learn matplotlib
```

Run the Script
Click the Play button in the top right of VS Code, or type this in the terminal:
```bash
    python main.py
```

