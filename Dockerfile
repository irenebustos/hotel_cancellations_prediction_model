# Use the AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.10

# Copy the requirements file and install dependencies
RUN pip install numpy==1.23.1
RUN pip install xgboost
RUN pip install pandas
RUN pip install xgboost
RUN pip install scikit-learn
RUN pip install requests
RUN pip install pandas

# Copy the Lambda function handler and model file into the container
COPY lambda_function.py .
COPY xgboost_model_booking_cancellation_smote.bin .

# Set the Lambda function handler
CMD ["lambda_function.lambda_handler"]
