import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model
model = joblib.load("model.pkl")

# Initialize FastAPI
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="ML API to predict employee attrition",
    version="1.0"
)


# Input data format
class EmployeeData(BaseModel):
    Department: str
    JobRole: str
    Salary: float
    YearsAtCompany: int
    Satisfaction: int


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Employee Attrition Prediction API is running"
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: EmployeeData):

    input_data = pd.DataFrame([{
        "Department": data.Department,
        "JobRole": data.JobRole,
        "Salary": data.Salary,
        "YearsAtCompany": data.YearsAtCompany,
        "Satisfaction": data.Satisfaction
    }])

    prediction = model.predict(input_data)[0]

    result = "Yes" if prediction == 1 else "No"

    return {
        "prediction": result,
        "message": (
            "Employee may leave the company"
            if prediction == 1
            else "Employee is likely to stay"
        )
    }