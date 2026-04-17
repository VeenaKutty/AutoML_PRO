# AutoML Pro: End-to-End Machine Learning Pipeline

## Overview

AutoML Pro is a modular, end-to-end machine learning system designed to automatically handle the full ML workflow for tabular datasets. The system detects the problem type (classification or regression), performs data preprocessing, compares multiple models, selects the best-performing model, and provides predictions through an interactive web application.

This project is built with a strong emphasis on modular design, scalability, and real-world usability.

---

## Features

* Automatic detection of problem type (classification or regression)
* Data preprocessing pipeline:

  * Missing value handling
  * Encoding of categorical variables
  * Feature scaling
* Outlier detection and removal
* Feature selection using statistical methods
* Model comparison using cross-validation
* Hyperparameter tuning using GridSearchCV
* Handling of imbalanced datasets using SMOTE
* Model leaderboard with performance comparison
* Model persistence using joblib
* Interactive Streamlit application with:

  * Dataset preview
  * Exploratory Data Analysis (EDA)
  * Model training interface
  * Prediction interface

---

## Project Structure

```
AutoML_Pro/
│
├── src/
│   ├── config_loader.py
│   ├── logger.py
│   ├── problem_type.py
│   ├── preprocessing.py
│   ├── outlier.py
│   ├── feature_selection.py
│   ├── model_selection.py
│   ├── trainer.py
│   ├── predictor.py
│
├── config/
│   └── config.yaml
│
├── artifacts/
│   ├── model.pkl
│   ├── metrics.json
│
├── logs/
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/VeenaKutty/AutoML_PRO.git
cd AutoML_PRO
```

### 2. Create a virtual environment

```
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```
venv\Scripts\activate
```

**macOS/Linux:**

```
source venv/bin/activate
```

### 4. Install dependencies

```
pip install -r requirements.txt
```

---

## Running the Application

### Run Streamlit app

```
streamlit run app.py
```

---

## Usage

1. Upload a CSV dataset using the interface
2. Select the target column
3. Explore dataset insights via EDA visualizations
4. Click "Train Model" to:

   * Automatically preprocess the data
   * Train multiple models
   * Compare performance
   * Select the best model
5. View the model leaderboard
6. Enter input values and generate predictions

---

## Configuration

The system behavior is controlled through the `config/config.yaml` file.

You can modify:

* Model hyperparameters
* Feature selection settings
* Cross-validation folds
* SMOTE usage

---

## Technologies Used

* Python
* pandas, numpy
* scikit-learn
* imbalanced-learn
* Streamlit
* matplotlib, seaborn
* joblib
* PyYAML

---

## Limitations

* Designed primarily for structured/tabular datasets
* Limited feature engineering capabilities
* Performance depends on dataset quality and size
* Not a replacement for advanced AutoML frameworks

---

## Future Improvements

* Model explainability (SHAP)
* Advanced feature engineering
* Support for text and image data
* Model deployment via API
* Improved UI/UX with multi-page navigation
* Automated imbalance detection and handling

---

## License

This project is open-source and available under the MIT License.

---

## Author

Developed as an end-to-end machine learning system for practical implementation and learning purposes.
