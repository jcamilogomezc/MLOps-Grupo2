# Taller 3: Nivel 2 - Desarrollo con Airflow

## Integrantes
* Edgar Cruz Martinez
* Juan Camilo Gomez Cano
* Germán Andrés Ospina Quintero

## Documentación del funcionamiento

En el siguiente video se presenta el funcionamiento del proyecto:

[![Mira el video en YouTube](https://img.youtube.com/vi/iqvIPvcs0GY/0.jpg)](https://www.youtube.com/watch?v=iqvIPvcs0GY)

---

Este proyecto implementa una arquitectura básica de MLOps que permite al equipo de desarrollo:

- Crear y entrenar modelos de machine learning en un entorno interactivo de Jupyter Notebook, desplegado mediante Docker y gestionado con el gestor de paquetes `uv`.
- Almacenar los modelos generados en una carpeta compartida (`models`) para su posterior consumo.
- Exponer una API desarrollada con FastAPI que permite seleccionar y utilizar los modelos entrenados para realizar inferencias.

## Estructura del Proyecto

- `jupyter/`: Contiene el Dockerfile y dependencias para levantar el servidor de Jupyter Lab.
- `models/`: Carpeta donde se almacenan los modelos generados por los notebooks y que serán consumidos por el API.
- `api/`: Código y Dockerfile para el servicio de inferencia con FastAPI.
- `docker-compose.yml`: Orquestador de los servicios (Jupyter y API).
- `README.md`: Este archivo.