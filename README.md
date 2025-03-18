# Consumer Complaint Sentiment Analysis with Random Forest

This repository contains a Machine Learning notebook that uses the **Random Forest** algorithm to predict consumer ratings (from 1 to 5) based on their comments and the status of their complaints (resolved or unresolved). The dataset used is from the Brazilian government's consumer complaint platform.

## Dataset Overview

The dataset, provided by the Brazilian government, includes consumer complaints with:
- **Complaint ID**, **Company**, **Date**, **Location**, **Status**, **Complaint Description**, **Company Response**, **Rating**, and **Consumer Comment**.

### Download the Dataset
Due to its large size, the dataset is hosted externally. You can download it from the following link:

[Download Dataset] (https://www.mediafire.com/file/yo5c4488n3gmf40/dados2025.json/file)

Place the downloaded file in the same directory of this repository.

## Techniques Used

1. **Data Preprocessing**:
   - Filtered out entries with no consumer comments.
   - Converted `status` to binary values (`Resolved = 1`, `Unresolved = 0`).
   - Transformed `rating` into integer values.

2. **Feature Engineering**:
   - Used **TF-IDF** to convert comments into numerical vectors.
   - Limited features to the top 500 words based on TF-IDF scores.
   - Combined TF-IDF features with `status`.

3. **Modeling**:
   - Applied **Random Forest** for classification.
   - Used **GridSearchCV** to find the best hyperparameters.
   - Evaluated the model using **accuracy** and **balanced accuracy**.

4. **Results**:
   - **Accuracy**: 80.76%
   - **Balanced Accuracy**: 38.25%
   - The model performs well for ratings 1 and 5 but struggles with 2, 3, and 4 due to class imbalance.

## Next Steps

1. Address class imbalance using **oversampling** or **undersampling**.
2. Experiment with other models like **XGBoost** or **Neural Networks**.
3. Use metrics like **F1-score** and **Precision** for better evaluation.

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
Dependencies:
  - Python 3.x
  - Libraries: pandas, nltk, scikit-learn, numpy
3. Download the dataset from the link above and place it together with the others files
4. Open the notebook codigo_ml_randomforest.ipynb and run the cells.
