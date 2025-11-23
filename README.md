Project Report
Customer Churn Prediction using SHAP

Student Name:Srimathi M
Institution: Cultus Skill Center

Project Summary

This project focuses on predicting telecom customer churn and understanding why customers decide to leave the service. A powerful machine learning model called XGBoost is used to classify whether a customer is likely to churn. However, instead of treating the model like a black box, this project includes SHAP (Shapley Additive Explanations) to clearly explain how each feature influences predictions.
 
The dataset includes important customer-related attributes such as:                                                                                                                                                                                                                                                                                                                                         

 Contract type
 Monthly charges
 Internet service
 Tenure (months with the company)
 Payment method
 Additional services (Tech support, Online security, etc.)
 

These factors help us identify which customers are more likely to churn.

Machine Learning Model

Model Used: XGBoost Classifier
Techniques Applied:

 Handling missing values and cleaning the dataset
 Encoding categorical features
 Feature scaling
 Class imbalance correction using scale_pos_weight
 Hyperparameter tuning using GridSearchCV
 Evaluation using ROC-AUC score to ensure model reliability

Goal: ROC-AUC score > 0.85
(High score achieved during model evaluation)

SHAP Interpretability Results

SHAP helps us understand:

Global Explanation

Most important features influencing churn:

Month-to-month contract
Short tenure (new customers)
High monthly charges
Fiber optic internet service

These show that new customers paying more without long-term commitment are more likely to leave.

 Local Explanation

Three customer profiles were analyzed:

 Customer Type      SHAP Finding                                
 
 High-risk Churner | High bill + Month-to-month + Low tenure     
 Low-risk Customer | 2-year contract + Long tenure + Lower bill  
 Borderline Case   | Mixed influences → needs targeted retention 

This helps the telecom company understand each customer's risk and take action before they leave.

Feature Interaction Effects

MonthlyCharges × Tenure:
  High monthly bill mainly increases churn for new users.

Fiber Optic × Contract Type:
  Fiber optic users with month-to-month contract show highest churn risk.

These patterns guide better pricing and service strategies.

 Business Recommendations

Based on SHAP insights, the company should:

 Offer discounts or starter benefits for customers with tenure  12 months
 Encourage month-to-month users to upgrade to longer contracts
 Reduce churn for high-charge customers with introductory pricing
 Promote bundles & loyalty rewards for fiber optic users
 Focus on borderline customers with personalized offers

These actions help reduce churn and boost customer loyalty.

 Conclusion

This project combines:

 High prediction performance
 Human-understandable explanations
 Clear business improvements

SHAP transforms a traditional churn prediction model into a strategic decision-support tool, helping the telecom company save revenue by retaining customers before they leave.


Just reply: **A**, **B**, or **Both** 😊
