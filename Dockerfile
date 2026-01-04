FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY model/ model/

EXPOSE 9696

# Production WSGI server
CMD ["gunicorn", "--bind=0.0.0.0:9696", "src.predict:app"]
