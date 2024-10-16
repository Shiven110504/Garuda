# Use an ARM-compatible base image with Python 3.5
FROM arm32v7/python:3.5

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow 1.x compatible with Python 3.5
RUN pip3 install tensorflow==1.14.0

# Install other Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Set environment variables for paths
ENV DATA_DIR=/Users/shiven/Desktop/HackHarvard/MSNET/ISBDA
ENV CHECKPOINT_DIR=/Users/shiven/Desktop/HackHarvard/MSNET/DATA
ENV OUTPUT_JSON=/Users/shiven/Desktop/HackHarvard/MSNET/output.json

# Command to run the inference script
CMD ["python3", "predict.py", \
     "--config", "DATA.BASEDIR=${DATA_DIR}", "MODE_FPN=True", \
     "--load", "${CHECKPOINT_DIR}", "--evaluate", "${OUTPUT_JSON}"]
