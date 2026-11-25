# Single container for the live demo: builds the React frontend, then serves it
# and the FastAPI inference API on port 7860 (the Hugging Face Spaces default).
# Model weights must be present at backend/app/model/densenet121_063A.keras at build time.

FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user backend/app ./app
COPY --chown=user --from=frontend /frontend/dist ./static
USER user
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
