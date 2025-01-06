import pickle
from flask import Flask, request, jsonify
import xgboost as xgb

# Load the model and DictVectorizer
model_file = 'xgboost_model_booking_cancellation_smote.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('cancellation_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data as JSON
    input_data = request.get_json()
    
    # Transform the input data using DictVectorizer
    X = dv.transform([input_data])
    features = list(dv.get_feature_names_out())
    
    # Create a DMatrix for prediction
    dmatrix = xgb.DMatrix(X, feature_names=features)
    
    # Predict days in shelter (single prediction value)
    y_pred = model.predict(dmatrix)[0]
    
    # Convert y_pred to a native Python float
    y_pred = float(y_pred)  # Ensuring it's a Python float

    result = {
        'predict_cancellation_booking': y_pred
    }
    
    # Return the result as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
