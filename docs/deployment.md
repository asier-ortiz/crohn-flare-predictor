# 🚀 Guía de Deployment

Instrucciones para desplegar el servicio ML en diferentes entornos.

## 🏃 Quick Start - Docker

La forma más fácil de desplegar:

```bash
# 1. Build
docker build -t crohn-ml-api:latest .

# 2. Run
docker run -d \
  -p 8001:8001 \
  -v $(pwd)/models:/app/models \
  --name crohn-ml \
  crohn-ml-api:latest

# 3. Verificar
curl http://localhost:8001/health
```

---

## 🐳 Docker Compose

### Archivo docker-compose.yml

```yaml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8001:8001"
    volumes:
      - ./models:/app/models
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - WORKERS=4
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
```

### Comandos

```bash
# Iniciar
docker-compose up -d

# Ver logs
docker-compose logs -f ml-api

# Reiniciar
docker-compose restart ml-api

# Parar
docker-compose down
```

---

## ☁️ Deployment en Cloud

### Opción 1: Railway.app (Más fácil)

1. Conecta tu repositorio GitHub
2. Railway detecta el Dockerfile automáticamente
3. Configura variables de entorno
4. Deploy automático en cada push

**Variables de entorno:**
```
ENVIRONMENT=production
LOG_LEVEL=INFO
WORKERS=4
PORT=8001
```

### Opción 2: DigitalOcean App Platform

1. Conecta GitHub repo
2. Selecciona Dockerfile
3. Configura:
   - **Port**: 8001
   - **Health Check**: `/health`
   - **Instancia**: Basic (1 GB RAM suficiente)

### Opción 3: AWS (ECS/Fargate)

**Requisitos:**
- AWS CLI configurado
- ECR repository creado

```bash
# 1. Build y push a ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.region.amazonaws.com
docker build -t crohn-ml-api .
docker tag crohn-ml-api:latest <account>.dkr.ecr.region.amazonaws.com/crohn-ml-api:latest
docker push <account>.dkr.ecr.region.amazonaws.com/crohn-ml-api:latest

# 2. Crear task definition (ver aws-task-definition.json)
# 3. Crear servicio en ECS
```

### Opción 4: Google Cloud Run

```bash
# 1. Build y push a GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/crohn-ml-api

# 2. Deploy
gcloud run deploy crohn-ml-api \
  --image gcr.io/PROJECT_ID/crohn-ml-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8001
```

---

## 🔧 Configuración de Producción

### Variables de Entorno

```bash
# Requeridas
ENVIRONMENT=production
LOG_LEVEL=INFO

# Opcionales
WORKERS=4  # Número de workers de uvicorn
PORT=8001
CORS_ORIGINS=https://tu-app-web.com,https://api.tu-app.com

# Modelo
MODEL_PATH=/app/models/crohn_predictor.pkl
MODEL_VERSION=1.0.0
```

### Montar el modelo ML

**Opción 1: Incluir en imagen Docker**
```dockerfile
COPY models/crohn_predictor.pkl /app/models/
```

**Opción 2: Volumen (recomendado)**
```bash
docker run -v /host/models:/app/models crohn-ml-api
```

**Opción 3: S3/GCS (para cloud)**
```python
# Descargar en startup
import boto3
s3 = boto3.client('s3')
s3.download_file('bucket', 'model.pkl', '/app/models/model.pkl')
```

---

## 📊 Monitoring

### Health Check

```bash
# El load balancer debe verificar cada 30s
curl http://api:8001/health
```

### Logs

```bash
# Docker
docker logs crohn-ml -f

# Con log aggregation (ELK, DataDog, etc.)
# Configurar en docker-compose.yml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### Métricas (Prometheus)

Añadir endpoint de métricas:
```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

---

## 🔄 CI/CD

### GitHub Actions

`.github/workflows/deploy.yml`:

```yaml
name: Deploy ML API

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t crohn-ml-api .

      - name: Run tests
        run: docker run crohn-ml-api pytest

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push crohn-ml-api:latest

      - name: Deploy to production
        run: |
          # Comando específico de tu plataforma
```

---

## 🔒 Seguridad

### 1. Añadir API Key (opcional)

```python
# api/config.py
api_key: str = "change-me-in-production"

# api/app.py
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.api_key:
        raise HTTPException(401, "Invalid API key")

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(...):
    ...
```

### 2. HTTPS (Obligatorio en producción)

- Railway/Cloud Run: Automático
- Servidor propio: Usar Nginx + Let's Encrypt

```nginx
server {
    listen 443 ssl;
    server_name api.tudominio.com;

    ssl_certificate /etc/letsencrypt/live/api.tudominio.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.tudominio.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
    }
}
```

### 3. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(...):
    ...
```

---

## 📦 Actualizar el Modelo

### Sin Downtime

```bash
# 1. Reemplazar archivo .pkl
docker cp new_model.pkl crohn-ml:/app/models/crohn_predictor.pkl

# 2. Reiniciar servicio
docker restart crohn-ml

# O con rolling update (múltiples instancias)
# Actualizar una instancia a la vez
```

### Con Docker

```bash
# 1. Build nueva imagen
docker build -t crohn-ml-api:v2 .

# 2. Blue-green deployment
docker run -d --name crohn-ml-v2 crohn-ml-api:v2

# 3. Switch load balancer a v2
# 4. Parar v1
docker stop crohn-ml
```

---

## 🔍 Troubleshooting

### Problema: Container no inicia

```bash
# Ver logs
docker logs crohn-ml

# Común: Modelo no encontrado
# Solución: Verificar volumen montado correctamente
```

### Problema: Health check falla

```bash
# Verificar que modelo está cargado
docker exec crohn-ml ls -la /app/models/

# Verificar logs de startup
docker logs crohn-ml | grep "model"
```

### Problema: Alto uso de memoria

```bash
# Reducir workers
docker run -e WORKERS=2 crohn-ml-api

# O aumentar RAM del container
docker run -m 2g crohn-ml-api
```

---

## 📋 Checklist de Deployment

- [ ] Variables de entorno configuradas
- [ ] Modelo ML disponible
- [ ] Health check funciona
- [ ] HTTPS configurado
- [ ] CORS configurado correctamente
- [ ] Logging configurado
- [ ] Monitoring configurado (opcional)
- [ ] CI/CD configurado (opcional)
- [ ] Documentación actualizada
- [ ] Equipo web tiene la nueva URL

---

## 💰 Costos Estimados

| Plataforma | Plan | Costo/mes |
|------------|------|-----------|
| Railway | Starter | $5-10 |
| DigitalOcean | Basic Droplet | $6 |
| AWS ECS | t3.small | ~$15 |
| Google Cloud Run | Pay-as-you-go | $5-20 |
| Heroku | Basic | $7 |

**Recomendación:** Railway o DigitalOcean para empezar.

---

## 📞 Soporte Post-Deployment

Una vez desplegado:
1. Notifica al equipo web de la nueva URL
2. Actualiza `ML_API_URL` en el backend web
3. Prueba integración end-to-end
4. Monitorea logs primeros días
