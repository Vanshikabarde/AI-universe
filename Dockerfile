FROM python:3.11-slim

# HuggingFace Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# HF Spaces expects the app to bind on port 7860 by default,
# but we expose 8000 and set PORT env var
ENV PORT=7860

CMD ["python", "server.py"]