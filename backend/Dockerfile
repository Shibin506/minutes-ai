FROM python:3.12-slim

WORKDIR /app
COPY backend/pyproject.toml backend/
COPY backend/minutes backend/minutes
RUN pip install --no-cache-dir ./backend
COPY samples samples

ENV MINUTES_DB_PATH=/data/minutes.db \
    MINUTES_SAMPLES_DIR=/app/samples
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "minutes.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
