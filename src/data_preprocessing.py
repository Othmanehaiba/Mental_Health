
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("data/raw/student_mental_health.csv")

# 2. Separate features and target
X = df.drop(columns=["Mental_Health_Risk", "Student_ID"])
y = df["Mental_Health_Risk"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Save training data
train_data = X_train.copy()
train_data["Mental_Health_Risk"] = y_train

train_data.to_csv(
    "data/processed/train.csv",
    index=False
)

# 5. Save testing data
test_data = X_test.copy()
test_data["Mental_Health_Risk"] = y_test

test_data.to_csv(
    "data/processed/test.csv",
    index=False
)

print("Data preprocessing completed!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
