# Usa una imagen base ligera de Python (la que has estado usando)
FROM python:3.9-slim

# Crea el directorio de trabajo
WORKDIR /app

# 1. Instala CURL de forma robusta
# El 'apt-get clean' y el 'apt-get update' se realizan en una sola línea.
# El -o Acquire::ForceIPv4=true a veces soluciona problemas de DNS en ciertas redes (como las VPNs).
RUN apt-get clean && apt-get update -o Acquire::ForceIPv4=true && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Copia los requisitos e instala las dependencias.
# Mantenemos el pip install para instalar scikit-learn con la versión correcta.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copia el código de la aplicación
COPY app/ /app/app/

# 4. Configuración final
EXPOSE 8000
CMD ["uvicorn", "app.api_alternative:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🚀 Pasos de Ejecución Final

Debes ejecutar la secuencia de comandos **completa** para asegurarte de que Docker use este nuevo `Dockerfile` y no el caché anterior.

1.  **Detener y Eliminar la Imagen Fallida:**

    ```bash
    # Detener el contenedor
    docker-compose -f docker-compose.alternative.yml down

    # Obtener el ID de la imagen local para eliminarla
    docker images

    # Eliminar la imagen (reemplaza <IMAGE_ID>)
    docker rmi <IMAGE_ID> -f
    ```

2.  **Reconstruir Forzadamente y Levantar:**

    ```bash
    # Reconstruir la imagen sin usar la caché (Esto forzará la nueva instalación de curl)
    docker-compose -f docker-compose.alternative.yml build --no-cache

    # Levantar el contenedor con la nueva imagen (sin -d para ver los logs iniciales de uvicorn)
    docker-compose -f docker-compose.alternative.yml up --force-recreate
    
