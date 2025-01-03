# README

This README was generated automatically from the notebook.

## Markdown Documentation
The database used for the model consists in a set of bookings from a hotel with a unique id called ¨boooking_id¨ from 2017 and 2018.

## Markdown Documentation
There are no missing values in any of the columns.

## Markdown Documentation
### Column Names and Data Types

- **Booking_ID (object)**: A unique identifier for each booking made at the hotel. It can be used for referencing a particular booking record.

- **no_of_adults (int64)**: The number of adults included in the booking. This value helps determine the size and requirements of the room or service.

- **no_of_children (int64)**: The number of children included in the booking. This value, along with the number of adults, may help decide room requirements or the need for additional services such as extra beds.

- **no_of_weekend_nights (int64)**: The number of weekend nights (typically Friday and Saturday nights) in the booking. This could indicate higher demand during weekends, which could affect the likelihood of cancellations.

- **no_of_week_nights (int64)**: The number of weeknights (typically Sunday to Thursday nights) in the booking. This can help to identify the pattern of booking durations and potentially different cancellation rates for weekdays versus weekends.

- **type_of_meal_plan (object)**: Describes the type of meal plan included in the booking (e.g., breakfast only, half board, full board, or all-inclusive). This may affect customer satisfaction and cancellation behavior.

- **required_car_parking_space (int64)**: Indicates whether the customer has requested parking space during their stay. It could reflect the customer's need for convenience or transportation and might be related to the cancellation decision.

- **room_type_reserved (object)**: Specifies the type of room the guest has reserved (e.g., single, double, suite). Room preferences can be linked to customer satisfaction, which might influence cancellation rates.

- **lead_time (int64)**: The number of days between the booking date and the scheduled arrival date. A longer lead time might suggest less likelihood of cancellation, while short-term bookings might be more prone to cancellation.

- **arrival_year (int64)**: The year in which the booking is scheduled to arrive. This can be useful for analyzing seasonal patterns and trends in cancellations over the years.

- **arrival_month (int64)**: The month of the scheduled arrival. It is useful for identifying seasonal trends in booking cancellations and can help account for periods with higher cancellation rates.

- **arrival_date (int64)**: The specific date of arrival (day of the month). This may help analyze cancellations during peak or off-peak days of the month.

- **market_segment_type (object)**: Describes the segment of the market from which the booking originated (e.g., direct, corporate, online travel agents, etc.). Different market segments could have varying cancellation rates, depending on their typical customer behavior.

- **repeated_guest (int64)**: A flag indicating whether the guest is a repeat customer (1 for repeat guests, 0 for first-time guests). Repeat guests may be less likely to cancel, as they have already established trust with the hotel.

- **no_of_previous_cancellations (int64)**: The number of previous bookings made by the guest that were cancelled. A higher number of previous cancellations could be a predictor for future cancellations.

- **no_of_previous_bookings_not_canceled (int64)**: The number of previous bookings made by the guest that were not cancelled. This can provide insight into the guest's general booking behavior and predict the likelihood of cancellation.

- **avg_price_per_room (float64)**: The average price per room booked by the customer. Higher-priced bookings may be less likely to be cancelled, as the cost is more substantial for the guest.

- **no_of_special_requests (int64)**: The number of special requests made by the guest (e.g., room preferences, extra beds, or other accommodations). A higher number of requests could indicate a higher likelihood of the booking being special to the guest and less likely to be cancelled.

- **booking_status (object)**: The status of the booking, indicating whether it was cancelled or not. This is the target variable for the machine learning model, representing the outcome we aim to predict (cancelled or not).

## Markdown Documentation
### New Columns Created

- **total_people**: The number of total people in the booking (adults & kids). This column represents the total occupancy for the booking.

- **price_per_adult** and **price_per_person**: Average price per night by adult or person in the booking (including kids). These columns provide a breakdown of the cost per individual in the booking.

- **has_previous_cancellations**: A flag indicating whether the user has previous cancellations. Instead of using the exact number of previous cancellations, this flag simply identifies if the guest has had cancellations before.

- **has_previous_bookings_not_cancelled**: A flag indicating whether the user has previous bookings that were not cancelled. This flag is used due to data limitations and replaces the exact number of previous bookings that were not cancelled.

- **total_nights**: The total amount of nights per booking. This column sums up the total nights the guest will stay at the hotel.

- **have_children**: A flag indicating whether the booking includes children. This flag replaces the exact number of children, simplifying the analysis.

## Markdown Documentation
#### Problem Description

The goal is to understand why 33% of all hotel bookings are canceled, which represents a significant impact on the business. Additionally, the task is to predict the likelihood of a booking being canceled over time, allowing the hotel to better estimate its capacity. This will enable the hotel to optimize booking availability by freeing up space for other customers when there is a high probability of cancellations.

