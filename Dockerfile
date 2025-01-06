# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install pipenv for dependency management
RUN pip install pipenv

# Create a directory in the container
WORKDIR /app

# Copy Pipfile and Pipfile.lock to the container to install dependencies
COPY Pipfile Pipfile.lock ./

# Install the dependencies with pipenv
RUN pipenv install --system --deploy

# Copy the remaining files (models, script, etc.) into the container
COPY predict.py xgboost_model_booking_cancellation_smote.bin ./

# Expose the port that Flask will run on (Gunicorn will bind to this port)
EXPOSE 9696

# Set the entry point to run the Flask app using Gunicorn
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
