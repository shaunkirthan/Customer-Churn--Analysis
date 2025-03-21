# Customer Churn Analysis

## 📌 Overview
This project focuses on analyzing and predicting **customer churn** using machine learning techniques. The dataset contains customer demographics, service details, and contract information, which are used to identify factors influencing churn. The primary objective is to develop a predictive model that can help businesses **proactively retain customers** by identifying potential churners.

## 📂 Project Structure
```
├── data/                     # Folder containing dataset(s)
├── notebooks/                # Jupyter Notebooks for EDA & Model Building
│   ├── customer_churn_analysis.ipynb  # Main analysis notebook
├── models/                   # Saved trained models
├── results/                  # Confusion matrices, precision-recall curves, and other visualizations
├── README.md                 # Project documentation (this file)
```

## 📊 Dataset
The dataset contains customer information, service subscriptions, and churn status. The target variable is:
- `Churn`: Binary classification (1 = Churn, 0 = No Churn)

### **Features**
- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Service Details**: Internet Service, Online Security, Online Backup, Streaming TV, Streaming Movies, etc.
- **Contract Details**: Contract Type, Payment Method, Tenure, Monthly Charges, Total Charges

## 🛠 Data Preprocessing
- **Handling Missing Values**: Imputed missing `TotalCharges` by converting strings to numeric and replacing nulls.
- **Encoding Categorical Variables**: Used **one-hot encoding** for features with more than two categories.
- **Scaling Numerical Features**: Standardized `MonthlyCharges` and `TotalCharges` for better model performance.
- **Handling Class Imbalance**: Addressed imbalance using **SMOTE** or `class_weight` in model training.

## 🏗 Model Building
### **Machine Learning Models Used**
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- Neural Network (TensorFlow/Keras)

### **Model Training Process**
1. **Split the dataset** into training (80%) and testing (20%).
2. **Feature engineering** applied to categorical and numerical columns.
3. **Hyperparameter tuning** performed using Grid Search & Randomized Search.
4. **Cross-validation** used to validate models' generalizability.

## 📈 Model Evaluation
### **Key Metrics**
- **Accuracy**: Measures overall correctness
- **Precision (Class 1 - Churn)**: Measures how many predicted churners are actual churners
- **Recall (Class 1 - Churn)**: Measures how many actual churners were correctly identified
- **F1-Score**: Balances precision and recall
- **ROC-AUC Score**: Measures model’s discrimination ability

### **Confusion Matrix Interpretation**
| Actual / Predicted | Predicted: No Churn (0) | Predicted: Churn (1) |
|--------------------|------------------------|----------------------|
| **Actual: No Churn (0)** | True Negatives (TN)  | False Positives (FP) |
| **Actual: Churn (1)** | False Negatives (FN)  | True Positives (TP) |

- **High False Negatives (FN)** → The model fails to identify churners → Leads to customer loss
- **High False Positives (FP)** → Predicts churners incorrectly → Leads to unnecessary retention offers

## 🔍 Insights & Business Impact
- **Tenure** and **contract type** are the most important factors in customer churn.
- Customers with **month-to-month contracts** are more likely to churn.
- **High monthly charges** increase churn probability.
- **Loyal customers (long tenure)** have lower churn rates.

## 📌 How to Use the Code
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/customer-churn-analysis.git
cd customer-churn-analysis
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Jupyter Notebook**
```bash
jupyter notebook
```
- Open `customer_churn_analysis.ipynb` and run all cells.

### **4️⃣ Train and Evaluate the Model**
If you want to re-train the model, run:
```python
python train_model.py
```

## 🔮 Future Improvements
- Implement **Deep Learning (LSTMs, CNNs)** for time-series customer behavior.
- Deploy the model using **Flask/Django API** for real-time predictions.
- Develop a **dashboard in Tableau/Power BI** to visualize churn trends.

## 📜 License
This project is open-source and available under the **MIT License**.

## 🙌 Acknowledgments
Thanks to open datasets and ML communities for contributing ideas and best practices in churn prediction!

