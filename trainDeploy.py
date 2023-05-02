import kserve
import argparse
import numpy as np
import joblib
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train and deploy a machine learning model')
parser.add_argument('--model_path', type=str, help='Path to save the trained model')
parser.add_argument('--data_path', type=str, help='Path to the data file')
parser.add_argument('--num_trees', type=int, default=100, help='Number of trees in the random forest')
args = parser.parse_args()

# Load the data
data = datasets.load_iris()
X = data.data
y = data.target

# Training the model
model = RandomForestClassifier(n_estimators=args.num_trees, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, args.model_path)

# Serving the model with Kserve
predictor = kserve.KServingPredictor(args.model_path)

# Testing the model
x_test = np.array([[5.1, 3.5, 1.4, 0.2]])
y_pred = predictor.predict(x_test)
print('Prediction:', y_pred)
