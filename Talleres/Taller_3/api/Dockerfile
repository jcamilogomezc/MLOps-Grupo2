# Dockerfile para el servicio web API
# Este Dockerfile configura un contenedor para ejecutar una aplicación web Python usando uv

# Usar la imagen base oficial de uv con Python 3.12 en Debian Bookworm slim
# uv es un gestor de paquetes y entornos virtuales rápido para Python
FROM astral/uv:python3.12-bookworm-slim

# Establecer el directorio de trabajo principal
WORKDIR /app

# Copiar todo el código fuente del proyecto al directorio web_service
COPY . /app/web_service/

# Cambiar al directorio donde está el código de la aplicación
WORKDIR /app/web_service

# Inicializar un nuevo proyecto uv en el directorio actual
# --no-readme evita la creación automática de un archivo README.md
RUN uv init --no-readme
    
# Instalar las dependencias del proyecto desde requirements.txt
# uv add -r lee el archivo requirements.txt e instala todas las dependencias listadas
RUN uv add -r requirements.txt

# Listar el contenido del directorio para verificar que los archivos se copiaron correctamente
# Útil para debugging durante la construcción de la imagen
RUN ls /app/web_service/

# Exponer el puerto 8989 para que la aplicación sea accesible desde fuera del contenedor
EXPOSE 8989

# Comando por defecto para ejecutar la aplicación
# uv run ejecuta uvicorn (servidor ASGI) con la aplicación main:app
# --host 0.0.0.0 permite conexiones desde cualquier IP
# --port 8989 especifica el puerto donde se ejecutará la aplicación
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8989"]