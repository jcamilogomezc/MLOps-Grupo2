# Taller 1: Nivel 0 - Docker + FastAPI

## Documentación del funcionamiento

En el siguiente video se presenta el funcionamiento de la API para la predicción de las especies. Se hace uso de *Swagger* para la demostración.

[![Mira el video en YouTube](https://img.youtube.com/vi/uuWJKZ3NUfw/0.jpg)](https://www.youtube.com/watch?v=uuWJKZ3NUfw)

---

## Preparación de los datos y creación de los modelos

### Notebook de entrenamiento

El notebook (`Taller Pinguinos.ipynb`) entrena y evalúa los modelos para predecir las especies de pingüinos usando el dataset **Palmer Penguins**.

#### Flujo de trabajo

1. **Carga de datos**
   - Lectura del dataset `penguins.csv`.
   - Vista previa y estadísticas.

2. **Exploración de datos (EDA)**
   - Gráficos con `seaborn` para visualizar distribuciones y relaciones.
   - Limpieza y manejo de valores nulos.

3. **Preprocesamiento**
   - Codificación de variables categóricas.
   - Escalado de variables numéricas.
   - Uso del transformador personalizado `CategoryCleaner`.

4. **Entrenamiento de modelos**
   - Modelos probados:
     - `MLP` (Multi-Layer Perceptron)
     - `HGB` (Histogram Gradient Boosting)
     - `LogReg` (Regresión Logística)
   - Uso de `Pipeline` para integrar preprocesamiento y modelo.

5. **Evaluación**
   - Métricas de precisión (`accuracy`).
   - Matrices de confusión.
   - Comparación de resultados entre modelos.

6. **Guardado de modelos**
   - Exportación con `joblib`:
     - `penguins_pipeline_MLP.joblib`
     - `penguins_pipeline_HGB.joblib`
     - `penguins_pipeline_LogReg.joblib`
     - `penguins_target_encoder.joblib`

### Transformador personalizado

Este código (`custom_transformers.py`) crea un **transformador personalizado** (`CategoryCleaner`) para scikit-learn que limpia las columnas categóricas.

#### Funcionamiento general
- Convierte textos a **minúsculas**.
- Elimina **espacios** extra.
- Rellena valores vacíos con `"desconocido"`.
- Reemplaza `"nan"` por `"desconocido"`.

#### Flujo de trabajo
1. **Detecta columnas categóricas** (tipo texto) o usa las indicadas.
2. **Limpia cada columna** con reglas de normalización.
3. **Devuelve** un DataFrame limpio y listo para el modelo.

---

## Docker

En la carpeta `docker` se encuentra el Dockerfile que crea una imagen ligera para la aplicación de Python que permite hacer inferencia de los modelos. Se emplea **FastAPI** para el manejo de peticiones y respuestas, además se hace uso de **Uvicorn** como servidor.

### Descripción del Dockerfile

1. **Imagen base**
   - `python:3.10-slim` En su versión *slim* para mantener la imagen ligera.

2. **Variables de entorno**
   - `PYTHONUNBUFFERED=1`: Logs en tiempo real.
   - `PIP_NO_CACHE_DIR=1`: Evita guardar caché de `pip`.
   - `MODEL_DIR=/models`: Define la carpeta para modelos.
   - `APP_HOME=/app`: Define la carpeta raíz de la app.

3. **Instalación de dependencias del sistema**
   - Instala `build-essential` para compilar paquetes Python.
   - `rm -rf /var/lib/apt/lists/*` limpia la caché de `apt` para reducir tamaño.

4. **Seguridad**
   - Crea usuario `appuser` y evita ejecutar como root.

5. **Instalación de dependencias Python**
   - Copia `requirements.txt` e instala librerías necesarias.

6. **Copia de código y modelos**
   - Copia carpeta `app/` y `models/` al contenedor.
   - Asigna permisos a `appuser`.

7. **Configuración del servidor**
   - Expone el puerto `8989`.
   - Ejecuta el servidor:
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port 8989
     ```

---

## FastAPI

Se hace uso de FastAPI como servidor. A continuación se explica brevemente el funcionamiento de los códigos fundamentales para el proyecto, los cuales se encuentran dentro de la carpeta `app`.

### Main

Este código (`main.py`) implementa una API con **FastAPI** para utilizar los modelos de predicción.

#### Funcionalidad principal
- **Carga de modelos**: Se usa `load_models()` para obtener un diccionario de modelos y un codificador de clases.
- **Pandas**: Se convierten los datos recibidos en DataFrames para pasarlos a los modelos.

#### Métodos 

1. `GET /health`
- Verifica que el servicio está activo.
- Devuelve:
  - Estado (`ok`).
  - Lista de modelos disponibles.
  - Clases que los modelos pueden predecir.

2. `POST /predict`
- Recibe:
  - Nombre del modelo a usar.
  - Lista de observaciones (datos de pingüinos).
- Realiza:
  - Validación de que el modelo existe.
  - Predicción con el modelo correspondiente.
  - Conversión de IDs numéricos a etiquetas de especie.
- Devuelve:
  - Nombre del modelo.
  - Lista de predicciones.
  - Cantidad de predicciones realizadas.

### Schemas

Este código (`schemas.py`) define modelos de datos con **Pydantic** para validar y estructurar la información.

#### Clases

1. `PenguinFeatures`
- Representa las características de un pingüino:
  - `bill_length_mm` (float)
  - `bill_depth_mm` (float)
  - `flipper_length_mm` (float)
  - `body_mass_g` (float)
  - `island` (str)
  - `sex` (str)

2. `PredictRequest`
- Formato de la solicitud de predicción:
  - `model_name`: `"MLP"`, `"HGB"` o `"LogReg"` (por defecto `"LogReg"`).
  - `items`: lista de `PenguinFeatures`.

3. `PredictResponse`
- Formato de la respuesta de la predicción:
  - `model_name` (str)
  - `predictions`: lista de especies (List[str])
  - `count`: cantidad de predicciones (int)

### Models Loader

Este código (`models_loader.py`) carga los modelos entrenados y el codificador de etiquetas para la API.

#### Funciones y comportamiento

- **Registro de clases personalizadas**
  - Añade `CategoryCleaner` al espacio de nombres `__main__` para que los modelos se puedan cargar sin errores.

- **Directorio de modelos**
  - Usa la variable de entorno `MODEL_DIR` o `/models` por defecto.

- **Archivos esperados**
  - Modelos:
    - `penguins_pipeline_MLP.joblib`
    - `penguins_pipeline_HGB.joblib`
    - `penguins_pipeline_LogReg.joblib`
  - Codificador de clases:
    - `penguins_target_encoder.joblib`

- **Función `load_models()`**
  1. Busca y carga cada modelo desde `MODEL_DIR`.
  2. Verifica que todos los archivos existan, si no, lanza `FileNotFoundError`.
  3. Carga el codificador de etiquetas (`target_encoder`).
  4. Devuelve:
     - Diccionario de modelos (`nombre → objeto modelo`).
     - Objeto codificador de clases.
---

## Ejecutar el proyecto

### Ejecutar localmente

```shell
python3 -m venv env
source env/bin/activate   
pip3 install -r requirements/requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 80
```

### Ejecutar localmente con docker
```shell
docker build -f docker/Dockerfile -t taller_1_image . && docker run --name taller_1_image -p 8000:80 taller_1_container
```