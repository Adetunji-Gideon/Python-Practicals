# Week 12: Data Analysis
# Iris Species Classification Using Machine Learning


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# 1. Load the Iris dataset
df = pd.read_csv("Iris.csv")

# 2. Display the first five rows
print("First five rows of the dataset:")
print(df.head())

# 3. Select input features and target variable
X = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

y = df["Species"]

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Standardize the input features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Create and train the machine learning model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="macro"
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro"
)

print("\nMODEL PERFORMANCE")
print("-----------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# 9. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
