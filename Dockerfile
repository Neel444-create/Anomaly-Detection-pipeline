# Lightweight runtime image for the FastAPI inference service.
FROM python:3.12-slim

WORKDIR /app

# Keep Python logs immediate and prevent bytecode files in the container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The prediction API needs its application code and the trained local artifact.
COPY api ./api
COPY src ./src
COPY models/best_model.joblib ./models/best_model.joblib
COPY models/best_model_metadata.json ./models/best_model_metadata.json

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
