## Customer Churn Prediction Using Machine Learning

In today's competitive business landscape, customer retention is paramount for sustainable growth and success. Our challenge is to develop a predictive model that can identify customers who are at risk of churning – discontinuing their use of our service. Customer churn can lead to a significant loss of revenue and a decline in market share. By leveraging machine learning techniques, we aim to build a model that can accurately predict whether a customer is likely to churn based on their historical usage behavior, demographic information, and subscription details. This predictive model will allow us to proactively target high-risk customers with personalized retention strategies, ultimately helping us enhance customer satisfaction, reduce churn rates, and optimize our business strategies. The goal is to create an effective solution that contributes to the long-term success of our company by fostering customer loyalty and engagement.





## 🧠 Project Summary

This project leverages machine learning to predict customer churn using the **SyriaTel Telecom** dataset. It includes thorough data exploration, model experimentation, and final recommendations to help stakeholders reduce churn and improve retention.

---

## 🎯 Business Objective

To help the telecom company **identify customers likely to churn** so that retention strategies can be targeted effectively. Early detection of churn-prone users can dramatically reduce customer loss and maximize lifetime value.

---

## 📁 Repository Contents

```
📦 Customer_Churn_Prediction/
├── 📊 churn_prediction_notebook.ipynb     # Main Jupyter notebook with code and visualizations
├── 📈 Output/                             # Folder for charts and saved visuals
│   └── churn_analysis_hd.png
│   └── roc_curves.png
│   └── confusion_matrices.png
├── 📜 README.md                           # Project overview and documentation
├── 📦 data/                               # Raw data (not uploaded if sensitive)
├── 📦 models/                             # Saved model files (optional)
├── 🧾 requirements.txt                    # Libraries used in the project
```

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights from the dataset:

* **Churn Rate:** \~14% of customers have churned.
* **Customer Service Calls:** Higher churn observed in users who contact customer service more frequently.
* **Geographical Trends:** Certain states show higher churn rates (see figure below).

### 🗺️ Churn Rate by State

![State Churn](Output/state_churn_hd.png)

* Some states such as NJ and TX show unusually high churn rates.
* Geographic factors can influence service experience and competition levels.

---

## 🤖 Machine Learning Models Used


### 🔬 Evaluation Visuals

#### ROC Curves

![ROC Curves](Output/roc_curves.png)

#### Confusion Matrices

![Confusion Matrices](Output/confusion_matrices.png)

---

## 📌 Key Takeaways

* **XGBoost** outperformed all other models and is the best fit for predicting churn.
* Geographic factors (states) are **valuable** and should not be ignored in modeling.
* Imbalanced data required careful handling (SMOTE improved recall).

---

## 📢 Recommendations

1. **Target high-churn states** (e.g., NJ, TX) with tailored retention campaigns.
2. **Monitor customers making >3 service calls** — prioritize them for intervention.
3. Integrate this ML model into the **CRM system** for real-time churn alerts.

---

## 🚀 How to Reproduce

1. Clone the repository.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the notebook: `churn_prediction_notebook.ipynb`
4. Review outputs in the `Output/` folder.

---

## 💼 For Stakeholders

This project gives SyriaTel a **data-driven strategy to reduce churn** and **maximize customer lifetime value**. It empowers proactive decisions and retention actions that directly affect the bottom line.

> 📥 All visuals are saved in `Output/` folder in HD and can be used in presentations or dashboards.

---

## ✨ Author

Ray | Data Scientist | Nairobi, Kenya 🌍

---

Let me know if you’d like a PowerPoint template or interactive dashboard based on these results.
