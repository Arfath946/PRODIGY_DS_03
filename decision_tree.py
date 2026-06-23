import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("bank.csv.csv", sep=";")

# Convert all text columns to numeric
le = LabelEncoder()

for col in df.columns:
    if str(df[col].dtype) in ["object", "str", "string"]:
        df[col] = le.fit_transform(df[col].astype(str))

print("Data Types After Encoding:")
print(df.dtypes)

# Features and target
X = df.drop("y", axis=1)
y = df["y"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Decision Tree Visualization
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree Classifier")
plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
plt.show()