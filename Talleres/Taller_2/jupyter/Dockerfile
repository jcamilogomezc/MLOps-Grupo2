FROM astral/uv:python3.12-bookworm-slim

WORKDIR /app

COPY . /app/ml_service/

WORKDIR /app/ml_service

RUN uv init --no-readme && rm -f main.py 
# Eliminar el archivo main.py que se crea por defecto
    
RUN uv add -r requirements.txt

RUN ls /app/ml_service/

EXPOSE 8888

CMD ["uv", "run", "jupyter", "lab", "--notebook-dir=/app", "--ip=0.0.0.0", "--allow-root", "--port=8888", "--no-browser"]