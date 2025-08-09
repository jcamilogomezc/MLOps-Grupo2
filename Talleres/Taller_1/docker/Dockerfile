FROM python:3.9

COPY app/ /app/
COPY requirements/ /requirements/

RUN pip3 install --no-cache-dir --upgrade -r requirements/requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]