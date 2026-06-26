
#  Customer Churn Prediction & Customer Segmentation System

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn and segments customers into meaningful business categories using interactive visualizations built with Streamlit.

---

##  Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Losing existing customers is often more expensive than acquiring new ones.

This project uses Machine Learning to predict customer churn and provides business insights through customer segmentation and interactive dashboards.

The system helps businesses:

- Predict customers who are likely to leave
- Understand churn behaviour
- Visualize customer statistics
- Identify high-risk customer groups
- Support retention strategies

---

#  Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Pickle

---

#  Project Structure

```
Customer-Churn-Prediction-System
│
├── app
│   └── app.py
│
├── model
│   ├── churn_model.pkl
│   └── encoder_columns.pkl
│
├── notebook
│   └── churn_prediction.ipynb
│
├── dataset
│   └── Telco_customer_churn.xlsx
│
├── requirements.txt
│
└── README.md
```

---

#  Dataset

The project uses the **Telco Customer Churn Dataset**.

The dataset contains customer information including:

- Customer demographics
- Contract type
- Internet service
- Monthly charges
- Total charges
- Tenure
- Payment method
- Additional telecom services
- Churn status

Target Variable:

```
Churn Value
```

- 1 → Customer Left
- 0 → Customer Stayed

---

#  Exploratory Data Analysis

Before training the model, the dataset was analyzed to understand customer behaviour.

The following analyses were performed:

- Missing value detection
- Duplicate checking
- Data cleaning
- Feature distributions
- Churn distribution
- Correlation analysis
- Contract-wise churn
- Tenure analysis
- Monthly charge analysis

Several visualizations were created using Seaborn and Matplotlib.

---

#  Data Preprocessing

The following preprocessing steps were applied:

- Removed unnecessary columns
- Handled missing values
- Converted Total Charges into numeric values
- One-Hot Encoding for categorical variables
- Feature selection
- Train-Test Split

Categorical variables were encoded using:

```
pd.get_dummies()
```

---

#  Machine Learning Model

The project uses:

## Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees to improve prediction accuracy and reduce overfitting.

Reasons for choosing Random Forest:

- Handles categorical data well
- Works well on tabular datasets
- Reduces overfitting
- High prediction accuracy
- Robust against noisy data

The trained model was saved using:

```python
pickle.dump(model, open("churn_model.pkl","wb"))
```

---

#  Model Performance

| Metric | Score |
|---------|---------|
| Accuracy | **74.38%** |
| Precision | **53.16%** |
| Recall | **82.00%** |
| F1 Score | **64.50%** |

### Confusion Matrix

| | Predicted No | Predicted Yes |
|------|-------------|--------------|
| Actual No | 720 | 289 |
| Actual Yes | 72 | 328 |

### Interpretation

- Accuracy of **74.38%** indicates good overall prediction performance.
- Recall of **82%** means the model successfully identifies most customers who are likely to churn.
- Precision of **53.16%** indicates that over half of the predicted churners actually churn.
- High Recall makes the model suitable for customer retention campaigns where identifying potential churners is more important than avoiding false alarms.

---

#  Customer Segmentation

To provide business insights, customers were grouped into three meaningful categories based on tenure and contract information.

###  Loyal Customers

- Long tenure
- Annual/Two-year contracts
- Low churn probability

###  Regular Customers

- Average tenure
- Moderate monthly charges
- Average churn behaviour

###  At-Risk Customers

- Short tenure
- Month-to-month contracts
- High churn probability

These segments help businesses prioritize customer retention efforts.

---

#  Dashboard Features

The project includes an interactive dashboard built using **Streamlit**.

### 1️ Overview

Displays:

- Total Customers
- Churned Customers
- Churn Rate
- Average Monthly Charges

Charts include:

- Churn by Contract
- Churn Distribution
- Tenure Distribution

---

### 2️ Customer Segments

Displays:

- Segment Summary
- Customer Distribution
- Churn Rate by Segment
- Monthly Charges vs Tenure Visualization

---

### 3️ Churn Prediction

Users can enter customer details such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Contract Type
- Internet Service
- Monthly Charges
- Total Charges
- Payment Method
- Tech Support

The application predicts:

- Churn Probability
- Risk Category
- Key Risk Factors
- Recommended Business Action

---

#  Future Improvements

- SHAP Explainable AI
- Customer Lifetime Value prediction
- Real-time database integration
- Customer login system
- Automated retention recommendations

---

#  Learning Outcomes

Through this project, I learned:

- Data Cleaning and Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- One-Hot Encoding
- Machine Learning Model Development
- Model Evaluation
- Customer Segmentation
- Interactive Dashboard Development using Streamlit
- Business Interpretation of Machine Learning Results

---

#  Conclusion

This project demonstrates how Machine Learning can help telecom companies proactively identify customers who are likely to churn.

By combining predictive analytics with customer segmentation and interactive dashboards, businesses can make informed decisions, improve customer retention strategies, and reduce revenue loss.
