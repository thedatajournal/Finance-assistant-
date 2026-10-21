FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image small
RUN pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY run_*.sh ./

CMD ["bash", "run_realtime_prod.sh"]
