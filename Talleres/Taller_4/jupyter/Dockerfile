FROM astral/uv:python3.12-bookworm-slim

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN uv init --no-readme && rm -f main.py
    
RUN uv add -r requirements.txt

EXPOSE 8888

CMD ["uv", "run", "jupyter", "lab", "--notebook-dir=/app", "--ip=0.0.0.0", "--allow-root", "--port=8888", "--no-browser"]