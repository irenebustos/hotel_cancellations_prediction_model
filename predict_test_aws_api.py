
import requests
import json

# Simplified event creation
event_data = {
    "type_of_meal_plan": "meal_plan_1",
    "room_type_reserved": "room_type_6",
    "market_segment_type": "online",
    "wday": "Wednesday",
    "no_of_weekend_nights": 1,
    "no_of_week_nights": 1,
    "required_car_parking_space": 0,
    "lead_time": 31,
    "repeated_guest": 0,
    "price_per_person": 81,
    "avg_price_per_room": 162.35,
    "no_of_special_requests": 0,
    "total_nights": 2,
    "arrival_month": 6,
    "no_of_adults": 2,
    "have_children": 0
}

# Convert the dictionary into the required 'body' format for the Lambda event
event = {"body": json.dumps(event_data)}

# Define the URL for the Lambda API endpoint
url = 'https://qjuy4kq6u5.execute-api.eu-west-1.amazonaws.com/test/predict'

# Send the POST request with the event
result = requests.post(url, json=event).json()

# Print the result
print(result)
