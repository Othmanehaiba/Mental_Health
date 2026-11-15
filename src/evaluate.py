import pandas as pd
import joblib
import json

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load test data
test_data = pd.read_csv("data/processed/test.csv")

X_test = test_data.drop(columns=["Mental_Health_Risk"])
y_test = test_data["Mental_Health_Risk"]

# 2. Load trained model
model = joblib.load("models/best_model.pkl")

# 3. Make predictions
y_pred = model.predict(X_test)

# 4. Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

print(classification_report(y_test, y_pred))

# 5. Save metrics
with open("reports/metrics.json", "w") as file:
    json.dump(
        {"accuracy": accuracy},
        file,
        indent=4
    )

with open("reports/classification_report.json", "w") as file:
    json.dump(report, file, indent=4)

# 6. Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("reports/confusion_matrix.png")
plt.show()

print("Evaluation completed!")
