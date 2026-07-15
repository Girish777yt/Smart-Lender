# ER Diagram

Entities:
- User (Credit Officer / Applicant)
- Applicant_Profile
- Credit_History
- Loan_Application
- Model
- Prediction_Result

Relationships (brief):
- User → Loan_Application (1-to-many)
- Applicant_Profile → Loan_Application (1-to-many)
- Credit_History → Applicant_Profile (1-to-1)
- Loan_Application → Prediction_Result (1-to-1)
- Model → Prediction_Result (1-to-many)
