FROM python:3.11-slim

# Configuración básica
ENV PYTHONPATH="$PYTHONPATH:$PWD"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto (debe coincidir con docker-compose)
EXPOSE 8000

# Comando para ejecutar FastAPI con uvicorn
CMD ["uvicorn", "src.controller.inference:app", "--host", "0.0.0.0", "--port", "8000"]