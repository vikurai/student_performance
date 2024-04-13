from flask import Flask, render_template, request
import pickle
import pandas as pd  # Import pandas for DataFrame manipulation

# Load the model
with open('student_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    hours_studied = float(request.form['hours_studied'])
    previous_scores = float(request.form['previous_scores'])
    sleep_hours = float(request.form['sleep_hours'])
    sample_papers = float(request.form['sample_papers'])
    
    # Define the feature names as a list
    feature_names = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']
    
    # Create a DataFrame with the input data and the feature names
    input_data = pd.DataFrame([[hours_studied, previous_scores, sleep_hours, sample_papers]], columns=feature_names)
    
    # Make a prediction using the model
    prediction = model.predict(input_data)
    
    # Render the result page with the prediction
    return render_template('result.html', prediction=prediction[0])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
