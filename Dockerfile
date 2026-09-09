FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for some ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the local package
RUN pip install -e .

# Expose the port (Hugging Face Spaces uses 7860, standard Docker uses 8000)
# We will use an environment variable for flexibility
ENV PORT=7860
EXPOSE $PORT

# Give execution rights to the startup script
RUN chmod +x start.sh

# Start the application
CMD ["./start.sh"]
