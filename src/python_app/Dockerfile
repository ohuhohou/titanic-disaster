# Python container for the Titanic logistic regression.
# Build from the REPO ROOT:  docker build -t titanic-python -f src/python_app/Dockerfile .
FROM python:3.11-slim

WORKDIR /app

# Install pinned dependencies first so this layer is cached between code changes.
COPY src/python_app/requirements.txt ./src/python_app/requirements.txt
RUN pip install --no-cache-dir -r src/python_app/requirements.txt

# Copy the code (the data is NOT baked into the image - it is mounted at run time).
COPY src/python_app/ ./src/python_app/

# Make Python print immediately so output streams to the terminal.
ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/src/data

CMD ["python", "src/python_app/titanic_model.py"]
