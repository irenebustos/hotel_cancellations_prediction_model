import requests
import json

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
data = {'type_of_meal_plan': 'meal_plan_1',
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

# Wrap the data in the "event" structure expected by Lambda
data = {"body": json.dumps(data)}

result = requests.post(url, json=data).json()
print(result)