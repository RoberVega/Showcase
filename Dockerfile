# Use the official Python image as the base image
FROM python:3.8

# Install system dependencies required for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the Docker image
WORKDIR /app

# Copy the entire current directory (Showcase) into the Docker image
COPY . /app

# Install pipenv
RUN pip install pipenv

# Install project dependencies using pipenv and system deploy prevents
# from creating a virtual environment within docker
RUN pipenv install --system --deploy

# Open port for Jupyter Notebook
EXPOSE 8888

# Open port for MLflow UI
EXPOSE 5000

# Open port for Flask
EXPOSE 9696

# Set environment variables for Jupyter Notebook (optional)
ENV JUPYTER_ALLOW_INSECURE_WRITES=1
ENV JUPYTER_TOKEN=my_secret_token

# Set environment variable for MLflow (optional)
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Deploy the flask app with gunicorn
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]

# Start Jupyter Notebook and MLflow UI (you may need to adjust the commands based on your specific setup)
CMD ["pipenv", "run", "bash", "-c", "pipenv shell && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=$JUPYTER_TOKEN --NotebookApp.allow_origin='*'"]


