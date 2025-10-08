# Import the librares
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
# Let us load the public iris dataset from sklearn
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
target_names = iris.target_names

# Train a simple RandomForest model     
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("Model training completed.")

joblib.dump(model, 'iris_model.joblib')
joblib.dump(target_names, 'iris_target_names.joblib')

print("Model and target names saved successfully as 'iris_model.joblib' and 'iris_target_names.joblib'.")
