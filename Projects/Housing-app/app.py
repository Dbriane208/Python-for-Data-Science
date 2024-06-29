from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_filename = 'Linear_regression_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the House Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Convert JSON data to DataFrame
    try:
        input_data = pd.DataFrame([data])
    except ValueError:
        return jsonify({'error': 'Invalid input data'}), 400

    # Make predictions
    try:
        predictions = model.predict(input_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Convert prediction to standard Python float
    prediction = float(predictions[0])

    # Return predictions as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
