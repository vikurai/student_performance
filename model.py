import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load your data
data = pd.read_csv("student_performance.csv")

# Define the features and target
features = data[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']]
target = data['Performance Index']

# Train a Random Forest Regressor model
model = RandomForestRegressor()
model.fit(features, target)

# Save the model using pickle
with open('student_model.pkl', 'wb') as file:
    pickle.dump(model, file)
