import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
n = 200

departments = ["Sales", "IT", "HR", "Finance", "Operations"]

roles = {
    "Sales": ["Sales Manager", "Sales Executive", "Account Manager"],
    "IT": ["Developer", "Tester", "IT Manager", "Data Analyst"],
    "HR": ["Recruiter", "HR Executive", "HR Manager"],
    "Finance": ["Accountant", "Financial Analyst", "Finance Manager"],
    "Operations": ["Coordinator", "Operations Executive", "Operations Manager"]
}

department = rng.choice(departments, n)
jobrole = [rng.choice(roles[d]) for d in department]

years = rng.integers(0, 16, n)
satisfaction = rng.integers(1, 6, n)

salary = [
    int(rng.integers(25000, 95001) + years[i] * 1200)
    for i in range(n)
]

score = 1 - 0.65 * (satisfaction - 3) - 0.12 * years
prob = 1 / (1 + np.exp(-score))

attrition = np.where(rng.random(n) < prob, "Yes", "No")

df = pd.DataFrame({
    "Department": department,
    "JobRole": jobrole,
    "Salary": salary,
    "YearsAtCompany": years,
    "Satisfaction": satisfaction,
    "Attrition": attrition
})

df.to_csv("employee.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("Dataset shape:", df.shape)


