FROM astral/uv:python3.12-bookworm-slim

WORKDIR /app

COPY . .

RUN uv init
    
RUN uv add -r requirements.txt

RUN ls ..

EXPOSE 8888

CMD ["uv", "run", "jupyter", "lab", "--notebook-dir=/app", "--ip=0.0.0.0", "--allow-root", "--port=8888", "--no-browser"]