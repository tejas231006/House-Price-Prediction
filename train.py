import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = "data/train.csv"
MODEL = "models/best_model.pkl"
FIG = "outputs/figures"
RESULTS = "outputs/results/model_comparison.csv"

os.makedirs("models", exist_ok=True)
os.makedirs(FIG, exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

df = pd.read_csv(DATA)

print("Dataset:", df.shape)

# EDA
plt.figure(figsize=(8, 5))
sns.histplot(df["SalePrice"], kde=True)
plt.title("House Price Distribution")
plt.tight_layout()
plt.savefig(f"{FIG}/price_distribution.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="sqft_living", y="SalePrice", alpha=0.4)
plt.title("Living Area vs Price")
plt.tight_layout()
plt.savefig(f"{FIG}/living_area_vs_price.png")
plt.close()

# EXACT features used by app.py
features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "yr_built"
]

X = df[features]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "KNN": Pipeline([
        ("scale", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=5))
    ]),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=20,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
}

results = []
fitted = {}

# Basic models
for name, model in models.items():

    print("Training:", name)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append([
        name,
        mean_absolute_error(y_test, pred),
        np.sqrt(mean_squared_error(y_test, pred)),
        r2_score(y_test, pred)
    ])

    fitted[name] = model

# Tune KNN
knn_grid = GridSearchCV(
    Pipeline([
        ("scale", StandardScaler()),
        ("model", KNeighborsRegressor())
    ]),
    {
        "model__n_neighbors": [5, 10, 15],
        "model__weights": ["uniform", "distance"]
    },
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

print("Tuning KNN...")
knn_grid.fit(X_train, y_train)

pred = knn_grid.predict(X_test)

results.append([
    "KNN Tuned",
    mean_absolute_error(y_test, pred),
    np.sqrt(mean_squared_error(y_test, pred)),
    r2_score(y_test, pred)
])

# Tune Random Forest
rf_grid = GridSearchCV(
    RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ),
    {
        "n_estimators": [100, 150],
        "max_depth": [15, 20]
    },
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

print("Tuning Random Forest...")
rf_grid.fit(X_train, y_train)

pred = rf_grid.predict(X_test)

results.append([
    "Random Forest Tuned",
    mean_absolute_error(y_test, pred),
    np.sqrt(mean_squared_error(y_test, pred)),
    r2_score(y_test, pred)
])

# Results
results = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2"]
).sort_values("RMSE")

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

results.to_csv(RESULTS, index=False)

# Select best model
best_name = results.iloc[0]["Model"]

if best_name == "KNN Tuned":
    best_model = knn_grid
elif best_name == "Random Forest Tuned":
    best_model = rf_grid
else:
    best_model = fitted[best_name]

joblib.dump(best_model, MODEL)

print("\nBest Model:", best_name)
print("Model saved:", MODEL)

# Model comparison graph
plt.figure(figsize=(9, 5))
sns.barplot(data=results, x="Model", y="R2")
plt.xticks(rotation=20)
plt.title("Model R² Comparison")
plt.tight_layout()
plt.savefig(f"{FIG}/model_comparison.png")
plt.close()

print("\nTraining complete!")