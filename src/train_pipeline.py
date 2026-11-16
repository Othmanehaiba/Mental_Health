import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

train_data = pd.read_csv("data/processed/train.csv")

X_train = train_data.drop(columns=["Mental_Health_Risk"])
y_train = train_data["Mental_Health_Risk"]

numerical_columns = X_train.select_dtypes(
    include="number"
).columns

categorical_columns = X_train.select_dtypes(
    exclude="number"
).columns

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

best_score = 0
best_model = None
best_model_name = ""

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="f1_macro"
    )

    mean_score = scores.mean()

    print(name, "F1 Macro:", round(mean_score, 4))

    if mean_score > best_score:
        best_score = mean_score
        best_model = pipeline
        best_model_name = name

best_model.fit(X_train, y_train)

joblib.dump(best_model, "models/best_model.pkl")

print("\nBest model:", best_model_name)
print("Best F1 Macro:", round(best_score, 4))
print("Model saved successfully!")
