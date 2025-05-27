# File: Dockerfile
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_USER appuser
ENV APP_HOME /home/$APP_USER/app

# Install system dependencies
# libsndfile1 is needed for the soundfile library
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN groupadd -r $APP_USER && useradd --no-log-init -r -g $APP_USER $APP_USER

# Create app directory
RUN mkdir -p $APP_HOME/model
WORKDIR $APP_HOME

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the model directory and its contents are owned by the app user
# This is important if the model is copied or generated in the image
# If model is generated at runtime by export_model.py inside container,
# ensure the script writes to a user-writable location or this is run after.
RUN chown -R $APP_USER:$APP_USER $APP_HOME

# Switch to non-root user
USER $APP_USER

# Expose port
EXPOSE 8000

# Run Uvicorn server with multiple workers
# The number of workers can be tuned. A common recommendation is (2 * number_of_cores) + 1
# For now, let's default to 2 workers. This can be overridden at runtime.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]