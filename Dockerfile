# What base image to use
FROM python:3.10-slim 

# Set the working directory in the container
WORKDIR /workspace

# Install and updates necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

# Install the dependencies listed in requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/workspace