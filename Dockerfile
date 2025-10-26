# Base image with Python and system tools
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy project metadata and install dependencies first (for layer caching)
COPY pyproject.toml requirements.txt ./

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY airbus_ship_detection/ airbus_ship_detection/
COPY model/ model/
COPY data/ data/

# Default command (can be overridden when running the container)
CMD ["python", "airbus_ship_detection/inference_script.py"]
