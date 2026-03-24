FROM python:3.11-slim

WORKDIR /app

# python:3.11-slim strips most system libraries to keep the image small.
# LightGBM requires libgomp1 (OpenMP runtime) to load its compiled C++ library at runtime.
# apt-get update refreshes the package list, install -y libgomp1 installs it without prompting,
# and rm -rf cleans up the package cache so it doesn't bloat the image.
# All three are chained with && into one RUN so they form a single Docker layer.
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached unless requirements change)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what the API needs
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
