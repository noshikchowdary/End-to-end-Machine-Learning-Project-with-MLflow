# End-to-end MLflow Project: Wine Quality Classification

## 1. Project Overview

This project implements an end-to-end machine learning pipeline using MLflow for model tracking and experiment management. It trains and compares RandomForest and XGBoost classifiers on the Wine Quality dataset, providing a robust, modular, and reproducible workflow for tabular classification tasks.

## 2. Features
- 📊 EDA notebook with insights
- 🛠️ Modular training pipeline
- 📈 MLflow tracking with metrics and artifacts
- 🤖 Dual-model training and comparison (RandomForest & XGBoost)
- 🐳 Docker support
- ⚙️ Config-driven architecture

## 3. Tech Stack
- Python
- Scikit-learn
- XGBoost
- MLflow
- Pandas
- Matplotlib
- Seaborn

## 4. Project Structure
```
End-to-end-Machine-Learning-Project-with-MLflow/
├── config/              # YAML configs for pipeline and parameters
├── notebooks/           # EDA and analysis notebooks
│   └── eda.ipynb
├── research/            # Experimentation and research notebooks
├── src/
│   └── mlProject/
│       ├── components/  # Modular pipeline components
│       ├── pipeline/    # Pipeline stage scripts
│       ├── config/      # Config management
│       ├── entity/      # Config/data entities
│       ├── utils/       # Utility functions
│       └── ...
├── static/              # Static assets (CSS, JS, images)
├── templates/           # HTML templates
├── tests/               # Test scripts
├── Dockerfile           # Docker support
├── main.py              # Pipeline entrypoint
├── app.py               # (Optional) Web app
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

## 5. Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/noshikchowdary/End-to-end-Machine-Learning-Project-with-MLflow.git
   cd End-to-end-Machine-Learning-Project-with-MLflow
   ```
2. **Create a virtual environment and install requirements**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. **Run MLflow UI**
   ```bash
   mlflow ui
   # Visit http://localhost:5000 to track experiments
   ```
4. **Run the pipeline**
   ```bash
   python main.py
   ```
5. **Track runs and compare models in MLflow UI**
   - Open [http://localhost:5000](http://localhost:5000) in your browser.

## 6. Sample MLflow Output Screenshot
*TODO: Add screenshot of MLflow UI showing model runs and metrics.*

## 7. Next Steps
- Add Streamlit dashboard for interactive model exploration
- Deploy best model via FastAPI
- Integrate with MLflow Model Registry

## 8. Credits / Ownership
**Custom-built and extended by Noshik Chirumamilla**


