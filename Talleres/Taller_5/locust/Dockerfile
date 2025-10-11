FROM python:3.11-slim

WORKDIR /locust

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8089

CMD ["locust", "-f", "main.py"]