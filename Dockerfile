# Use an official Python runtime as a parent image (slim versions are smaller)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements first to cache the pip install step
COPY requirements.txt .

# Install Python dependencies (PyTorch is huge, this step takes a minute)
# --no-cache-dir keeps the Docker image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and the trained model
COPY app/ ./app/
COPY models/ ./models/

# Expose the port our FastAPI app runs on
EXPOSE 9000

# Command to run the application when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]