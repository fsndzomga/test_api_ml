from flask import Flask, request, jsonify
import pickle
import statsmodels.api as sm

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the Flask application
app = Flask(__name__)

# Define a route for the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json()

    # Make predictions with the model
    X = sm.add_constant(data['X'])
    y_pred = model.predict(X)

    # Return the predictions as a JSON response
    return jsonify({'predictions': list(y_pred)})

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
