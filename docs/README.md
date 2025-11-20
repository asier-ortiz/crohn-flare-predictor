# 📚 Documentación del Proyecto

Bienvenido a la documentación del servicio ML para predicción de brotes de enfermedad de Crohn.

## 📖 Índice de Documentación

### Para Desarrolladores del Equipo

- **[Guía de Integración](INTEGRATION.md)** - Cómo consumir esta API desde la aplicación web
- **[Referencia de API](API_REFERENCE.md)** - Documentación completa de endpoints

### Para Desarrollo ML (Mi parte)

- **[Arquitectura](ARCHITECTURE.md)** - Decisiones de diseño y estructura del proyecto
- **[Guía de Desarrollo](DEVELOPMENT.md)** - Setup local y flujo de trabajo
- **[Deployment](DEPLOYMENT.md)** - Cómo desplegar el servicio

## 🎯 ¿Qué es este proyecto?

Este es un **servicio ML independiente** (microservicio) que expone una API REST para predicción de brotes de enfermedad de Crohn basado en síntomas diarios.

### Responsabilidades

**Este servicio ML se encarga de:**
- ✅ Entrenar y mantener modelos de machine learning
- ✅ Exponer predicciones vía API REST
- ✅ Análisis de tendencias temporales
- ✅ Predicciones por lotes

**Este servicio NO se encarga de:**
- ❌ Gestión de usuarios (login, registro)
- ❌ Almacenamiento de datos de pacientes
- ❌ Frontend/UI
- ❌ Base de datos

## 🏗️ Arquitectura del Proyecto

```
crohn-flare-predictor/          # Este proyecto (ML API)
├── api/                        # Endpoints FastAPI
│   ├── app.py                 # Aplicación principal
│   ├── schemas.py             # Validación de datos
│   └── config.py              # Configuración
├── src/                        # Código ML
│   ├── model.py               # Modelos de ML
│   ├── preprocessing.py       # Preprocesamiento
│   └── feature_engineering.py # Features
├── models/                     # Modelos entrenados (.pkl)
├── notebooks/                  # Análisis exploratorio
├── docs/                       # Esta documentación
└── tests/                      # Tests unitarios
```

## 🚀 Quick Start

### Para desarrolladores del equipo web

Si solo necesitas consumir la API:

```bash
# 1. Asegúrate de que el servicio esté corriendo
curl http://localhost:8001/health

# 2. Lee la guía de integración
docs/INTEGRATION.md

# 3. Explora la documentación interactiva
http://localhost:8001/docs
```

### Para desarrollo ML

```bash
# 1. Clonar y setup
git clone <repo>
cd crohn-flare-predictor
uv sync --group dev --group notebooks

# 2. Iniciar servicio
make serve

# 3. Ver documentación de desarrollo
docs/DEVELOPMENT.md
```

## 📞 Contacto y Soporte

Si tienes problemas con la API ML, contacta conmigo.

Para issues con la aplicación web (backend/frontend), consulta con el equipo de desarrollo web.

## 📄 Licencia

MIT License - Ver archivo LICENSE para detalles.
