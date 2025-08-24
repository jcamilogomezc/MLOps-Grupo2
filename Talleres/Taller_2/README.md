# Taller 2: Nivel 1 - Desarrollo en Contenedores

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

## Requisitos

- Docker y Docker Compose instalados en el sistema.

## Uso Rápido

1. Clona el repositorio y navega a la carpeta del proyecto.
2. Levanta los servicios con:
	```bash
	docker-compose -f docker-compose.yml up --build
	```
3. Accede a Jupyter Lab en [http://localhost:8888](http://localhost:8888) usando el token que aparece en `docker-compose.yml`
4. Los modelos generados en los notebooks se guardarán en la carpeta `models` y estarán disponibles para el API.

## Contribuciones
Si quieres contribuir en el proyecto genera una nueva rama tomando como base `main`!
