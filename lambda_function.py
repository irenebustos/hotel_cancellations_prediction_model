import pickle
import json
import xgboost as xgb

# Load the model and DictVectorizer
model_file = 'xgboost_model_booking_cancellation_smote.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

def predict(input_data):
    # Transform the input data using DictVectorizer
    X = dv.transform([input_data])
    features = list(dv.get_feature_names_out())
    
    # Create a DMatrix for prediction
    dmatrix = xgb.DMatrix(X, feature_names=features)
    
    # Predict days in shelter (single prediction value)
    y_pred = model.predict(dmatrix)[0]
    
    # Convert y_pred to a native Python float
    y_pred = float(y_pred)  # Ensuring it's a Python float
    # in the result, if y_pred is >=0.5, it means the booking will be cancelled therefor say ¨Highly probable to be cancelled¨and otherwise ¨there is no risk of cancellation¨
    result = {
        'predict_cancellation_booking': 'Highly probable to be cancelled' if y_pred >= 0.5 else 'There is no risk of cancellation'
        
    }
    
    # result = {
    #     'predict_cancellation_booking': y_pred
    # }
    
    return result

def lambda_handler(event, context):
    try:
        # Extract input data from the event
        input_data = json.loads(event['body'])
        
        # Perform prediction
        result = predict(input_data)
        
        # Return the result
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
