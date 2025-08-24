# Dockerfile para crear un contenedor con Jupyter Lab
# Este contenedor proporciona un entorno de desarrollo para machine learning
# con Python 3.12 y gestión de dependencias mediante uv

# Usar la imagen base de astral/uv con Python 3.12 en Debian Bookworm
# Esta imagen incluye uv (gestor de paquetes rápido para Python) preinstalado
FROM astral/uv:python3.12-bookworm-slim

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install build-essential && \
    rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo principal
WORKDIR /app

# Copiar todos los archivos del directorio actual al contenedor
# Los archivos se copian a /app/ml_service/ para mantener una estructura organizada
COPY . /app/ml_service/

# Cambiar al directorio donde se copiaron los archivos del proyecto
WORKDIR /app/ml_service

# Inicializar un nuevo proyecto Python con uv
# --no-readme: no crear el archivo README.md por defecto
RUN uv init --no-readme && rm -f main.py
# Eliminar el archivo main.py que se crea por defecto
    
# Instalar las dependencias especificadas en requirements.txt
# uv add -r lee el archivo requirements.txt e instala todas las dependencias listadas
RUN uv add -r requirements.txt

# Listar el contenido del directorio para verificar que los archivos se copiaron correctamente
RUN ls /app/ml_service/

# Exponer el puerto 8888 para acceder a Jupyter Lab desde el navegador
EXPOSE 8888

# Comando por defecto para ejecutar Jupyter Lab
# --notebook-dir=/app: directorio donde se guardarán los notebooks
# --ip=0.0.0.0: permitir conexiones desde cualquier IP
# --allow-root: permitir ejecutar Jupyter como usuario root (necesario en contenedores)
# --port=8888: puerto en el que se ejecutará Jupyter Lab
# --no-browser: no abrir automáticamente el navegador
CMD ["uv", "run", "jupyter", "lab", "--notebook-dir=/app", "--ip=0.0.0.0", "--allow-root", "--port=8888", "--no-browser"]