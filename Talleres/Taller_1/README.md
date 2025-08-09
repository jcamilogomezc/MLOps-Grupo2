# Description

## Run project locally

```shell
python3 -m venv env
source env/bin/activate   
pip3 install -r requirements/requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 80
```

## Run project locally with docker
```shell
docker build -f docker/Dockerfile -t taller_1_image . && docker run --name taller_1_image -p 8000:80 taller_1_container
```