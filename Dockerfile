# OncoLens Streamlit app.
#   docker build -t oncolens .
#   docker run --rm -p 8501:8501 oncolens      # then open http://localhost:8501
FROM python:3.12-slim

# libgomp1 is the OpenMP runtime that xgboost and scikit-learn need on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Rebuild the model and all results inside the image so the container is self-contained.
RUN python run_all.py

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
