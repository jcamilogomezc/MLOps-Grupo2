FROM astral/uv:python3.12-bookworm-slim

WORKDIR /app

COPY . /app/web_service/

WORKDIR /app/web_service

RUN uv init --no-readme
    
RUN uv add -r requirements.txt

RUN ls /app/web_service/

EXPOSE 8989

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8989"]