import requests

url = 'http://localhost:9696/predict'

animal = {'type_of_meal_plan': 'meal_plan_1',
    'room_type_reserved': 'room_type_6',
    'market_segment_type': 'online',
    'wday': 'Wednesday',
    'no_of_weekend_nights': 1,
    'no_of_week_nights': 1,
    'required_car_parking_space': 0,
    'lead_time': 31,
    'repeated_guest': 0,
    'price_per_person': 81.0,
    'avg_price_per_room': 162.35,
    'no_of_special_requests': 0,
    'total_nights': 2,
    'arrival_month': 6,
    'no_of_adults': 2,
    'have_children': 0
}

# Send POST request to the server
response = requests.post(url, json=animal)

# Check if the response status code is OK (200)
if response.status_code == 200:
    try:
        # Try to parse the response as JSON
        response_json = response.json()
        
        # Extract the probability of cancellation from the response
        probability_cancellation = response_json.get('predict_cancellation_booking')
        
        if probability_cancellation is not None and isinstance(probability_cancellation, (int, float)):
            probability_cancellation = round(probability_cancellation, 2)  # Round to 2 decimal places
            if probability_cancellation > 0.5:
                print('It is highly likely that this booking will be cancelled.')
            else:
                print('It is unlikely that this booking will be cancelled.')
        else:
            print("Error: Invalid response from the API. Expected a numerical prediction.")
    
    except ValueError:
        # Handle case where response is not a valid JSON
        print("Error: The response content is not valid JSON.")
else:
    print(f"Error: Received unexpected status code {response.status_code}. Response: {response.text}")
