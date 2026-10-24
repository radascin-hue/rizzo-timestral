FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir fastapi "uvicorn[standard]" git+https://github.com/mistralai/timestral.git

COPY app.py .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
