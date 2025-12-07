FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./
COPY sample_data.csv ./

EXPOSE 5000
CMD ["python", "n8n_api.py"]
