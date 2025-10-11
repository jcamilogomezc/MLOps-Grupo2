from fastapi import FastAPI

from src.service.training import lifespan

app = FastAPI(lifespan=lifespan)