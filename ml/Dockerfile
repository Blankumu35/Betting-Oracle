FROM python:3.10-slim

WORKDIR /app

# Install build tools and system libraries for scientific Python packages
RUN apt-get update && apt-get install -y gcc g++ build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel for better compatibility
RUN pip install --upgrade pip setuptools wheel

COPY ml/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/. .
COPY recommendation_engine/ ./recommendation_engine/

# Install dependencies for recommendation_engine
RUN pip install --no-cache-dir -r recommendation_engine/requirements.txt

EXPOSE 8001

CMD ["uvicorn", "predict_fixtures:app", "--host", "0.0.0.0", "--port", "8001"] 