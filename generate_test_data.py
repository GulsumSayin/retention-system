import numpy as np
import pandas as pd

np.random.seed(99)
n = 100

# customerID
customer_ids = [f"TEST-{str(i+1).zfill(4)}" for i in range(n)]

# gender
gender = np.random.choice(["Male", "Female"], size=n, p=[0.50, 0.50])

# SeniorCitizen
senior_citizen = np.random.choice([0, 1], size=n, p=[0.80, 0.20])

# Partner
partner = np.random.choice(["Yes", "No"], size=n, p=[0.50, 0.50])

# Dependents
dependents = np.random.choice(["Yes", "No"], size=n, p=[0.30, 0.70])

# tenure: 30% 1-12, 30% 13-36, 40% 37-72
tenure_group = np.random.choice([0, 1, 2], size=n, p=[0.30, 0.30, 0.40])
tenure = np.where(
    tenure_group == 0,
    np.random.randint(1, 13, size=n),
    np.where(
        tenure_group == 1,
        np.random.randint(13, 37, size=n),
        np.random.randint(37, 73, size=n)
    )
)

# PhoneService
phone_service = np.random.choice(["Yes", "No"], size=n, p=[0.90, 0.10])

# MultipleLines
multiple_lines = np.where(
    phone_service == "No",
    "No phone service",
    np.where(
        np.random.random(n) < 0.45,
        "Yes",
        "No"
    )
)

# InternetService: seniors more likely DSL or No
internet_service = []
for i in range(n):
    if senior_citizen[i] == 1:
        # seniors: ~30% Fiber, ~40% DSL, ~30% No
        svc = np.random.choice(["Fiber optic", "DSL", "No"], p=[0.30, 0.40, 0.30])
    else:
        svc = np.random.choice(["Fiber optic", "DSL", "No"], p=[0.45, 0.35, 0.20])
    internet_service.append(svc)
internet_service = np.array(internet_service)

def internet_dependent_field(internet_arr, p_yes, is_fiber_boost=False):
    result = []
    for i in range(n):
        svc = internet_arr[i]
        if svc == "No":
            result.append("No internet service")
        else:
            prob = p_yes
            if is_fiber_boost and svc == "Fiber optic":
                prob = min(p_yes + 0.15, 0.95)
            val = np.random.choice(["Yes", "No"], p=[prob, 1 - prob])
            result.append(val)
    return np.array(result)

online_security    = internet_dependent_field(internet_service, 0.35)
online_backup      = internet_dependent_field(internet_service, 0.40)
device_protection  = internet_dependent_field(internet_service, 0.35)
tech_support       = internet_dependent_field(internet_service, 0.35)
streaming_tv       = internet_dependent_field(internet_service, 0.40, is_fiber_boost=True)
streaming_movies   = internet_dependent_field(internet_service, 0.40, is_fiber_boost=True)

# Contract: high monthly + short tenure → more Month-to-month
# We'll assign contract first loosely, then adjust below after charges
contract_base = np.random.choice(
    ["Month-to-month", "One year", "Two year"],
    size=n,
    p=[0.45, 0.25, 0.30]
)

# PaperlessBilling
paperless_billing = np.random.choice(["Yes", "No"], size=n, p=[0.60, 0.40])

# PaymentMethod
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    size=n,
    p=[0.33, 0.22, 0.22, 0.23]
)

# MonthlyCharges
def calc_monthly_charge(i):
    svc = internet_service[i]
    if svc == "No":
        base = np.random.uniform(20, 35)
    elif svc == "DSL":
        base = np.random.uniform(45, 75)
        # add ~5 per added service
        extras = sum([
            online_security[i] == "Yes",
            online_backup[i] == "Yes",
            device_protection[i] == "Yes",
            tech_support[i] == "Yes",
            streaming_tv[i] == "Yes",
            streaming_movies[i] == "Yes"
        ])
        base += extras * np.random.uniform(3, 7)
    else:  # Fiber optic
        base = np.random.uniform(70, 110)
        extras = sum([
            online_security[i] == "Yes",
            online_backup[i] == "Yes",
            device_protection[i] == "Yes",
            tech_support[i] == "Yes",
            streaming_tv[i] == "Yes",
            streaming_movies[i] == "Yes"
        ])
        base += extras * np.random.uniform(3, 7)
    return round(base, 2)

monthly_charges = np.array([calc_monthly_charge(i) for i in range(n)])

# Contract adjustment: high charges + short tenure → Month-to-month more likely
contract = []
for i in range(n):
    if monthly_charges[i] > 75 and tenure[i] <= 12:
        # Override to month-to-month with 70% probability
        if np.random.random() < 0.70:
            contract.append("Month-to-month")
        else:
            contract.append(contract_base[i])
    else:
        contract.append(contract_base[i])
contract = np.array(contract)

# TotalCharges: MonthlyCharges * tenure ± 5% noise
# ~2 rows get " " (missing)
noise = np.random.uniform(0.95, 1.05, size=n)
total_charges_numeric = np.round(monthly_charges * tenure * noise, 2)

# Pick 2 random indices for missing TotalCharges
missing_idx = np.random.choice(n, size=2, replace=False)
total_charges = [
    " " if i in missing_idx else str(total_charges_numeric[i])
    for i in range(n)
]

df = pd.DataFrame({
    "customerID":       customer_ids,
    "gender":           gender,
    "SeniorCitizen":    senior_citizen,
    "Partner":          partner,
    "Dependents":       dependents,
    "tenure":           tenure,
    "PhoneService":     phone_service,
    "MultipleLines":    multiple_lines,
    "InternetService":  internet_service,
    "OnlineSecurity":   online_security,
    "OnlineBackup":     online_backup,
    "DeviceProtection": device_protection,
    "TechSupport":      tech_support,
    "StreamingTV":      streaming_tv,
    "StreamingMovies":  streaming_movies,
    "Contract":         contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod":    payment_method,
    "MonthlyCharges":   monthly_charges,
    "TotalCharges":     total_charges,
})

output_path = r"C:\Users\glsms\Desktop\retention_system\data\test_100.csv"
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}\n")

print("=== First 5 rows ===")
print(df.head().to_string())
print()

print("=== Shape ===")
print(df.shape)
print()

print("=== Contract value counts ===")
print(df["Contract"].value_counts())
print()

print("=== InternetService value counts ===")
print(df["InternetService"].value_counts())
print()

print("=== MonthlyCharges describe() ===")
print(df["MonthlyCharges"].describe())
print()

print(f"Missing TotalCharges rows (space string): {sum(1 for v in total_charges if v == ' ')}")
print(f"Missing TotalCharges at indices: {sorted(missing_idx.tolist())}")
