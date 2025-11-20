import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import skops.io as sio
import os

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Prepare features and target
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Create ML Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
pipe.fit(X_train, y_train)

# Make predictions
predictions = pipe.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='macro')

print(f"Accuracy: {accuracy:.2%}, F1 Score: {f1:.2f}")

# Save metrics
with open("Results/metrics.txt", "w") as f:
    f.write(f"\nAccuracy = {accuracy:.2f}, F1 Score = {f1:.2f}")

# Create and save confusion matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)
disp.plot(cmap='Blues')
plt.title("Iris Classification - Confusion Matrix")
plt.savefig("Results/confusion_matrix.png", dpi=120, bbox_inches='tight')

# Save model using skops
sio.dump(pipe, "Model/iris_pipeline.skops")