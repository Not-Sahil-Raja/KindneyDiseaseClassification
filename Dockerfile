FROM python:3.10.12-slim-buster

# Install necessary packages
RUN apt-get update -y && apt-get install -y awscli



# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory to store the model
RUN mkdir -p /model
RUN python download_model.py

# Expose the port the app runs on
EXPOSE 8080

# Run the application with Gunicorn and Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]