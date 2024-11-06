FROM python:3.8-slim-buster

# Install necessary packages
RUN apt-get update -y && apt-get install -y awscli

# # Create a writable directory for logs
# RUN mkdir -p /tmp/logs && chmod -R 777 /tmp/logs

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Run the application with Gunicorn and Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]