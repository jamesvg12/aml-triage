FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# LightGBM needs the OpenMP runtime, which the slim image lacks
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# dependency layer first: cached unless requirements change
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml .
COPY src/ src/
COPY config/ config/
RUN pip install --no-cache-dir --no-deps .

# data, model and reports are mounted at run time, never baked in
ENTRYPOINT ["aml-score", "--config", "/app/config/config.yaml"]